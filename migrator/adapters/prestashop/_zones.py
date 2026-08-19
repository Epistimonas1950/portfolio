"""PrestaShop → OpenCart zone-code mapping.

PrestaShop's Greek state table stores the 52 pre-2011 prefectures (νομοί) —
the old administrative units before the Kallikratis reform consolidated them
into 13 modern regions (περιφέρειες). OpenCart's stock GR zone list uses
the 13 modern regions only, so the migration is a many-to-one rollup.

The map is keyed by the iso_code stored in psxm_state (NOT canonical
ISO 3166-2 — this PrestaShop install ships its own numbering where e.g.
GR-61 is Πιερίας/Pieria and GR-63 is Φλώρινας/Florina, which doesn't match
the now-deprecated ISO 3166-2 GR-NN prefecture codes). Values are the OC
zone codes from a stock `oc_zone` table for country GR.
"""
from __future__ import annotations


GR_PREFECTURE_TO_REGION: dict[str, str] = {
    # Attica (AT)
    "GR-A1": "AT",  # Αττικής

    # Central Macedonia (CM)
    "GR-53": "CM",  # Ημαθίας (Imathia)
    "GR-54": "CM",  # Θεσσαλονίκης (Thessaloniki)
    "GR-57": "CM",  # Κιλκίς (Kilkis)
    "GR-59": "CM",  # Πέλλας (Pella)
    "GR-61": "CM",  # Πιερίας (Pieria)
    "GR-62": "CM",  # Σερρών (Serres)
    "GR-64": "CM",  # Χαλκιδικής (Chalkidiki)
    "GR-69": "CM",  # Άγιο Όρος (Mount Athos — autonomous, geographically in CM)

    # East Macedonia and Thrace (EM)
    "GR-52": "EM",  # Δράμας (Drama)
    "GR-55": "EM",  # Καβάλας (Kavala)
    "GR-71": "EM",  # Έβρου (Evros)
    "GR-72": "EM",  # Ξάνθης (Xanthi)
    "GR-73": "EM",  # Ροδόπης (Rodopi)

    # West Macedonia (WM)
    "GR-51": "WM",  # Γρεβενών (Grevena)
    "GR-56": "WM",  # Καστοριάς (Kastoria)
    "GR-58": "WM",  # Κοζάνης (Kozani)
    "GR-63": "WM",  # Φλώρινας (Florina)

    # Epirus (EP)
    "GR-31": "EP",  # Άρτας (Arta)
    "GR-32": "EP",  # Θεσπρωτίας (Thesprotia)
    "GR-33": "EP",  # Ιωαννίνων (Ioannina)
    "GR-34": "EP",  # Πρέβεζας (Preveza)

    # Thessaly (TH)
    "GR-41": "TH",  # Καρδίτσας (Karditsa)
    "GR-42": "TH",  # Λάρισας (Larissa)
    "GR-43": "TH",  # Μαγνησίας (Magnesia)
    "GR-44": "TH",  # Τρικάλων (Trikala)

    # Ionian Islands (II)
    "GR-21": "II",  # Ζακύνθου (Zakynthos)
    "GR-22": "II",  # Κέρκυρας (Kerkyra / Corfu)
    "GR-23": "II",  # Κεφαλληνίας (Kefallonia)
    "GR-24": "II",  # Λευκάδας (Lefkada)

    # Central Greece / Sterea Ellada (CN)
    "GR-03": "CN",  # Βοιωτίας (Boeotia)
    "GR-04": "CN",  # Εύβοιας (Euboea)
    "GR-05": "CN",  # Ευρυτανίας (Evrytania)
    "GR-06": "CN",  # Φθιώτιδας (Phthiotis)
    "GR-07": "CN",  # Φωκίδας (Phocis)

    # Peloponnese (PP)
    "GR-11": "PP",  # Αργολίδας (Argolida)
    "GR-12": "PP",  # Αρκαδίας (Arkadia)
    "GR-15": "PP",  # Κορινθίας (Korinthia)
    "GR-16": "PP",  # Λακωνίας (Laconia)
    "GR-17": "PP",  # Μεσσηνίας (Messenia)

    # West Greece (WG)
    "GR-01": "WG",  # Αιτωλοακαρνανίας (Aetolia-Acarnania)
    "GR-13": "WG",  # Αχαΐας (Achaea)
    "GR-14": "WG",  # Ηλείας (Elis)

    # North Aegean (NA)
    "GR-83": "NA",  # Λέσβου (Lesbos)
    "GR-84": "NA",  # Σάμου (Samos)
    "GR-85": "NA",  # Χίου (Chios)

    # South Aegean (SA)
    "GR-81": "SA",  # Δωδεκανήσου (Dodecanese)
    "GR-82": "SA",  # Κυκλάδων (Cyclades)

    # Crete (CR)
    "GR-91": "CR",  # Ηρακλείου (Heraklion)
    "GR-92": "CR",  # Λασιθίου (Lasithi)
    "GR-93": "CR",  # Ρεθύμνου (Rethymno)
    "GR-94": "CR",  # Χανίων (Chania)
}


def ps_zone_to_oc(country_iso: str, ps_zone_iso: str) -> str:
    """Translate a PrestaShop state iso_code to an OpenCart zone code.

    Returns "" if no mapping exists (the writer will leave zone_id=0,
    same as before). Only Greek prefectures are mapped — other countries'
    states would need their own table added here.
    """
    if not ps_zone_iso:
        return ""
    if country_iso and country_iso.upper() == "GR":
        return GR_PREFECTURE_TO_REGION.get(ps_zone_iso.upper(), "")
    return ""
