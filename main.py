import os
import re
import json
import time
import hashlib
import logging
import urllib.parse
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
from bs4 import BeautifulSoup

# Intentar usar curl_cffi para evadir bloqueos Cloudflare/TLS; fallback a requests
try:
    from curl_cffi import requests as http_client
    USE_CURL_CFFI = True
except ImportError:
    import requests as http_client
    USE_CURL_CFFI = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("InmoRadar")

STATE_FILE = "seen_listings.json"


@dataclass
class Listing:
    id: str
    title: str
    price: Optional[float]
    url: str
    source: str
    location: str = "No especificada"
    description: str = ""
    # Atributos deducidos por el motor heurístico
    inferred_services: List[str] = field(default_factory=list)
    has_services: Optional[bool] = None
    real_estimated_price: Optional[float] = None
    inferred_capacity: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    matched_specs: List[str] = field(default_factory=list)
    matched_zone: Optional[str] = None
    score: int = 50
    profile_name: str = ""


class HeuristicAnalyzer:
    """Analiza texto libre en español para deducir características que los caseros no estructuran."""

    SERVICES_MAP = {
        "Agua": r"\b(agua|servicio de agua)\b",
        "Luz": r"\b(luz|electricidad|cfe)\b",
        "Internet": r"\b(internet|wifi|wi-fi|fibra\s*óptica)\b",
        "Gas": r"\b(gas)\b",
        "Servicios Grales.": r"\b(con servicios|servicios incluidos|incluye servicios|todo incluido)\b"
    }

    NO_SERVICES_REGEX = re.compile(
        r"\b(sin servicios|no incluye servicios|servicios aparte|más servicios|\+\s*servicios)\b",
        re.IGNORECASE
    )

    WARNING_PATTERNS = {
        "Solo mujeres/señoritas": r"\b(solo\s+mujeres|solo\s+señoritas|únicamente\s+mujeres|exclusivo\s+damas)\b",
        "Solo hombres": r"\b(solo\s+hombres|solo\s+caballeros|exclusivo\s+varones)\b",
        "Solo 1 persona": r"\b(solo\s+1\s+persona|solo\s+una\s+persona|individual\s+sin\s+excepci[oó]n|no\s+parejas)\b",
        "Pide Aval": r"\b(con\s+aval|aval\s+forzoso|aval\s+con\s+propiedad|fiador)\b",
        "Sin mascotas": r"\b(no\s+mascotas|sin\s+mascotas)\b",
    }

    COUPLE_FRIENDLY_REGEX = re.compile(
        r"\b(2\s*personas|dos\s*personas|pareja|parejas|roomies|compartir|2\s*rec[aá]maras|dos\s*rec[aá]maras)\b",
        re.IGNORECASE
    )

    @classmethod
    def enrich_and_score(cls, listing: Listing, profile: Dict[str, Any], global_cfg: Dict[str, Any]) -> bool:
        """
        Analiza el anuncio, infiere datos ocultos, calcula el score de 0 a 100
        y determina si pasa los filtros del perfil.
        """
        full_text = f"{listing.title} {listing.location} {listing.description}".lower()
        score = 50  # Base neutral

        # 1. Filtro de palabras excluyentes (Hard Filter)
        excluded_words = profile.get("palabras_excluyentes") or []
        for word in excluded_words:
            if word and word.lower() in full_text:
                return False

        # 2. Filtro Anti-Estafa y Presupuesto
        min_scam = global_cfg.get("precio_minimo_anti_estafa", 1000)
        if listing.price is not None and listing.price < min_scam:
            return False

        # 3. Inferencia de Servicios Incluidos
        if cls.NO_SERVICES_REGEX.search(full_text):
            listing.has_services = False
        else:
            for srv_name, pattern in cls.SERVICES_MAP.items():
                if re.search(pattern, full_text, re.IGNORECASE):
                    listing.inferred_services.append(srv_name)
            if listing.inferred_services:
                listing.has_services = True

        # Cálculo de Costo Real Mensual
        extra_srv_cost = global_cfg.get("costo_estimado_servicios_extra", 650)
        if listing.price is not None:
            if listing.has_services:
                listing.real_estimated_price = listing.price
            else:
                listing.real_estimated_price = listing.price + extra_srv_cost

        # 4. Validación de Presupuesto (con opción de Costo Real)
        p_min = profile.get("precio_min")
        p_max = profile.get("precio_max")
        use_real = profile.get("usar_costo_real_con_servicios", False)

        eval_price = listing.real_estimated_price if (use_real and listing.has_services is False) else listing.price

        if eval_price is not None:
            if p_min is not None and eval_price < p_min:
                return False
            if p_max is not None:
                if eval_price > p_max:
                    return False
                # Bono de score si está significativamente por debajo del presupuesto máximo
                ahorro_ratio = (p_max - eval_price) / p_max
                score += int(min(20, ahorro_ratio * 40))

        # 5. Evaluación de Servicios según preferencia del usuario
        pref_services = profile.get("con_servicios")
        strict_services = profile.get("servicios_estricto", False)

        if pref_services is True:
            if listing.has_services is True:
                score += 15
            elif listing.has_services is False:
                if strict_services:
                    return False
                score -= 10
            elif strict_services:
                return False
        elif pref_services is False and listing.has_services is False:
            score += 10

        # 6. Inferencia de Capacidad (Número de Personas)
        pref_personas = profile.get("personas")
        is_single_only = bool(re.search(cls.WARNING_PATTERNS["Solo 1 persona"], full_text, re.IGNORECASE))
        is_multi_friendly = bool(cls.COUPLE_FRIENDLY_REGEX.search(full_text))

        if is_single_only:
            listing.inferred_capacity = "Solo 1 persona"
        elif is_multi_friendly:
            listing.inferred_capacity = "Apto 2+ personas / Pareja"
        else:
            listing.inferred_capacity = "No especificado (Consultar)"

        if pref_personas is not None:
            if pref_personas >= 2:
                if is_single_only:
                    return False  # Descarta si pide explícitamente 1 sola persona
                if is_multi_friendly:
                    score += 15

        # 7. Evaluación de Zonas
        zonas = profile.get("zonas") or []
        zona_estricta = profile.get("zona_estricta", False)
        if zonas:
            for z in zonas:
                if z and z.lower() in full_text:
                    listing.matched_zone = z
                    score += 20
                    break
            if not listing.matched_zone:
                if zona_estricta:
                    return False
                score -= 5

        # 8. Especificaciones Deseadas
        specs = profile.get("especificaciones_deseadas") or []
        for spec in specs:
            if spec and spec.lower() in full_text:
                listing.matched_specs.append(spec)
                score += 6

        # 9. Detección de Advertencias / Banderas Rojas
        for warn_label, pattern in cls.WARNING_PATTERNS.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                listing.warnings.append(warn_label)

        listing.score = max(0, min(100, score))
        listing.profile_name = profile.get("nombre", "Búsqueda General")

        min_score = global_cfg.get("score_minimo_notificacion", 40)
        return listing.score >= min_score


class ScraperEngine:
    """Motor de extracción multi-fuente resistente a bloqueos en GitHub Actions."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
    }

    @classmethod
    def fetch(cls, url: str, params: Optional[dict] = None) -> Optional[str]:
        try:
            if USE_CURL_CFFI:
                resp = http_client.get(url, params=params, headers=cls.HEADERS, impersonate="chrome120", timeout=15)
            else:
                resp = http_client.get(url, params=params, headers=cls.HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.text
            logger.warning(f"Status {resp.status_code} al consultar {url}")
        except Exception as e:
            logger.warning(f"Error de red en {url}: {e}")
        return None

    @staticmethod
    def extract_price(text: str) -> Optional[float]:
        if not text:
            return None
        # Busca patrones tipo $4,500 o $ 4500 o 4,500 pesos
        matches = re.findall(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,6})", text)
        if matches:
            try:
                val = float(matches[0].replace(",", ""))
                if 800 <= val <= 50000:
                    return val
            except ValueError:
                pass
        return None

    @staticmethod
    def make_id(url: str) -> str:
        clean_url = url.split("?")[0].strip().lower()
        return hashlib.sha256(clean_url.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def search_mercadolibre(cls, profile: Dict[str, Any]) -> List[Listing]:
        """Extrae anuncios directamente de Mercado Libre Inmuebles México."""
        results = []
        ciudad = profile.get("ciudad_estado") or "mexico"
        tipo = profile.get("tipo_inmueble") or "inmuebles"
        zonas = profile.get("zonas") or [None]

        # Realiza búsqueda por las primeras 3 zonas o una general si no hay zonas
        search_targets = zonas[:3] if zonas else [None]

        for zona in search_targets:
            query_parts = ["renta", tipo, ciudad]
            if zona:
                query_parts.append(zona)
            slug = "-".join([urllib.parse.quote(p.lower().replace(" ", "-")) for p in query_parts if p])
            url = f"https://inmuebles.mercadolibre.com.mx/{slug}_NoIndex_True"

            html = cls.fetch(url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            cards = soup.select("li.ui-search-layout__item, div.ui-search-result__wrapper")

            for card in cards[:15]:
                link_tag = card.select_one("a.ui-search-link, a.poly-component__title")
                if not link_tag or not link_tag.get("href"):
                    continue

                href = link_tag["href"].split("#")[0].split("?")[0]
                title = link_tag.get_text(strip=True)

                price_tag = card.select_one("span.andes-money-amount__fraction")
                price = None
                if price_tag:
                    try:
                        price = float(price_tag.get_text(strip=True).replace(",", "").replace(".", ""))
                    except ValueError:
                        pass

                loc_tag = card.select_one("span.ui-search-item__location, span.poly-component__location")
                location = loc_tag.get_text(strip=True) if loc_tag else ciudad

                attrs = " ".join([el.get_text(" ", strip=True) for el in card.select("li.ui-search-card-attributes__attribute, span.poly-attributes-list__item")])

                results.append(Listing(
                    id=cls.make_id(href),
                    title=title,
                    price=price,
                    url=href,
                    source="Mercado Libre",
                    location=location,
                    description=f"{title} {location} {attrs}"
                ))
            time.sleep(1.2)
        return results

    @classmethod
    def search_meta_portals(cls, profile: Dict[str, Any]) -> List[Listing]:
        """
        Usa meta-indexación sobre Inmuebles24, Vivanuncios, Lamudi y Trovit
        para evitar bloqueos de Cloudflare en las IPs de GitHub Actions.
        """
        results = []
        ciudad = profile.get("ciudad_estado") or ""
        tipo = profile.get("tipo_inmueble") or "departamento"
        zonas = profile.get("zonas") or [""]
        personas = profile.get("personas")
        con_servicios = profile.get("con_servicios")

        portals = '(site:inmuebles24.com OR site:vivanuncios.com.mx OR site:lamudi.com.mx OR site:casasyterrenos.com)'
        zona_query = f'("{zonas[0]}")' if zonas and zonas[0] else ""
        srv_query = '"servicios"' if con_servicios else ""
        pers_query = f'"{personas} personas"' if personas and personas > 1 else ""

        dork = f'renta {tipo} {ciudad} {zona_query} {srv_query} {pers_query} {portals}'.strip()
        html = cls.fetch("https://html.duckduckgo.com/html/", params={"q": dork})
        if not html:
            return results

        soup = BeautifulSoup(html, "html.parser")
        for item in soup.select("div.result"):
            a_tag = item.select_one("a.result__a")
            snippet_tag = item.select_one("a.result__snippet")
            if not a_tag:
                continue

            raw_href = a_tag.get("href", "")
            # Limpiar redirección de DuckDuckGo si aplica
            if "uddg=" in raw_href:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                url = parsed.get("uddg", [raw_href])[0]
            else:
                url = raw_href

            if not url.startswith("http"):
                continue

            title = a_tag.get_text(" ", strip=True)
            snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
            combined_text = f"{title} {snippet}"

            price = cls.extract_price(combined_text)
            domain = urllib.parse.urlparse(url).netloc.replace("www.", "")

            results.append(Listing(
                id=cls.make_id(url),
                title=title[:120],
                price=price,
                url=url,
                source=domain,
                location=ciudad or "Ver en enlace",
                description=snippet
            ))

        return results


class DiscordNotifier:
    """Construye y envía Rich Embeds en Discord con enlaces a Google Maps y desglose inteligente."""

    @staticmethod
    def send_listing(webhook_url: str, item: Listing, moneda: str = "MXN") -> bool:
        # Color según calidad del Match Score
        if item.score >= 80:
            color = 0x2ECC71  # Verde Esmeralda (Match Excelente)
            badge = "🔥 OPORTUNIDAD TOP"
        elif item.score >= 60:
            color = 0x3498DB  # Azul (Buen Match)
            badge = "✅ BUEN CANDIDATO"
        else:
            color = 0xF39C12  # Naranja (Match Parcial / General)
            badge = "🔎 PARA REVISAR"

        price_str = f"${item.price:,.0f} {moneda}" if item.price else "No especificado en vista previa"
        if item.real_estimated_price and item.price and item.real_estimated_price != item.price:
            price_str += f"\n*(~${item.real_estimated_price:,.0f} est. con servicios)*"

        if item.has_services is True:
            srv_str = f"✅ Sí ({', '.join(item.inferred_services)})"
        elif item.has_services is False:
            srv_str = "❌ No incluidos (Se pagan aparte)"
        else:
            srv_str = "❓ No especifica (Preguntar al casero)"

        # Generar enlace automático de búsqueda en Google Maps
        maps_query = urllib.parse.quote(f"{item.matched_zone or ''} {item.location}".strip())
        maps_url = f"https://www.google.com/maps/search/?api=1&query={maps_query}"

        fields = [
            {"name": "💰 Precio Mensual", "value": price_str, "inline": True},
            {"name": "📍 Zona / Ubicación", "value": f"{item.location}\n[🗺️ Ver en Google Maps]({maps_url})", "inline": True},
            {"name": "📊 Match Score", "value": f"**{item.score}/100** ({badge})", "inline": True},
            {"name": "💡 Servicios", "value": srv_str, "inline": True},
            {"name": "👥 Capacidad Deducida", "value": item.inferred_capacity or "Consultar", "inline": True},
            {"name": "🌐 Portal Origen", "value": item.source, "inline": True},
        ]

        if item.matched_specs:
            fields.append({
                "name": "✨ Especificaciones Encontradas",
                "value": ", ".join([f"`{s}`" for s in item.matched_specs]),
                "inline": False
            })

        if item.warnings:
            fields.append({
                "name": "⚠️ Restricciones / Alertas Detectadas",
                "value": " • ".join(item.warnings),
                "inline": False
            })

        embed = {
            "username": "InmoRadar Bot",
            "embeds": [{
                "title": f"🏠 {item.title[:230]}",
                "url": item.url,
                "color": color,
                "description": (
                    f"**Perfil:** `{item.profile_name}`\n"
                    f"{item.description[:280]}...\n\n"
                    f"👉 **[ABRIR PUBLICACIÓN ORIGINAL]({item.url})**"
                ),
                "fields": fields,
                "footer": {
                    "text": f"ID: {item.id} • InmoRadar Automated Engine"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }

        try:
            resp = http_client.post(webhook_url, json=embed, timeout=10)
            if resp.status_code in (200, 204):
                return True
            elif resp.status_code == 429:
                # Respetar Rate Limit de Discord
                time.sleep(3)
                http_client.post(webhook_url, json=embed, timeout=10)
                return True
            logger.error(f"Error de Discord Webhook: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Excepción enviando a Discord: {e}")
        return False


def load_seen_ids() -> Dict[str, str]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_seen_ids(seen: Dict[str, str]):
    # Mantener solo los últimos 2,500 registros para que el archivo JSON nunca pese
    if len(seen) > 2500:
        keys = list(seen.keys())[-2500:]
        seen = {k: seen[k] for k in keys}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=2, ensure_ascii=False)


def apply_env_overrides(config: Dict[str, Any]):
    """Permite lanzar una búsqueda rápida desde la interfaz web de GitHub Actions sin editar el YAML."""
    override_city = os.getenv("OVERRIDE_CIUDAD", "").strip()
    if not override_city:
        return

    logger.info(f"Detectada ejecución manual rápida para: {override_city}")
    zonas_raw = os.getenv("OVERRIDE_ZONAS", "").strip()
    zonas = [z.strip() for z in zonas_raw.split(",") if z.strip()] if zonas_raw else []

    precio_max_raw = os.getenv("OVERRIDE_PRECIO_MAX", "").strip()
    precio_max = float(precio_max_raw) if precio_max_raw.isdigit() else None

    personas_raw = os.getenv("OVERRIDE_PERSONAS", "").strip()
    personas = int(personas_raw) if personas_raw.isdigit() else None

    config["perfiles_busqueda"] = [{
        "nombre": f"Búsqueda Manual ({override_city})",
        "activo": True,
        "ciudad_estado": override_city,
        "tipo_inmueble": None,
        "zonas": zonas,
        "zona_estricta": False,
        "precio_min": None,
        "precio_max": precio_max,
        "usar_costo_real_con_servicios": False,
        "personas": personas,
        "con_servicios": None,
        "servicios_estricto": False,
        "especificaciones_deseadas": [],
        "palabras_excluyentes": ["venta", "traspaso"]
    }]


def main():
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise ValueError
