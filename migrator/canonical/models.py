from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SourcePlatform(str, Enum):
    PRESTASHOP = "prestashop"
    MAGENTO1 = "magento1"
    MAGENTO2 = "magento2"
    WOOCOMMERCE = "woocommerce"


@dataclass
class ImageRef:
    source_path: str    # absolute path on local filesystem
    dest_filename: str  # relative path inside OpenCart image/catalog/
    is_cover: bool = False


@dataclass
class Manufacturer:
    source_id: int
    name: str
    sort_order: int = 0
    image: Optional[ImageRef] = None


@dataclass
class Category:
    source_id: int
    parent_source_id: Optional[int]
    sort_order: int
    status: bool
    names: dict[int, str] = field(default_factory=dict)           # oc_lang_id → name
    descriptions: dict[int, str] = field(default_factory=dict)    # oc_lang_id → html
    meta_titles: dict[int, str] = field(default_factory=dict)
    meta_descriptions: dict[int, str] = field(default_factory=dict)
    meta_keywords: dict[int, str] = field(default_factory=dict)
    image: Optional[ImageRef] = None


@dataclass
class Product:
    source_id: int
    sku: str
    model: str
    price: float
    quantity: int
    weight: float
    status: bool
    sort_order: int = 0
    manufacturer_source_id: Optional[int] = None
    ean: str = ""
    upc: str = ""
    isbn: str = ""
    mpn: str = ""
    minimum: int = 1
    width: float = 0.0
    height: float = 0.0
    depth: float = 0.0
    date_added: Optional[datetime] = None
    category_source_ids: list[int] = field(default_factory=list)
    names: dict[int, str] = field(default_factory=dict)
    descriptions: dict[int, str] = field(default_factory=dict)
    meta_titles: dict[int, str] = field(default_factory=dict)
    meta_descriptions: dict[int, str] = field(default_factory=dict)
    meta_keywords: dict[int, str] = field(default_factory=dict)
    images: list[ImageRef] = field(default_factory=list)


@dataclass
class Address:
    source_id: int
    customer_source_id: int
    firstname: str
    lastname: str
    company: str = ""
    address_1: str = ""
    address_2: str = ""
    city: str = ""
    postcode: str = ""
    country_iso: str = ""   # 2-char ISO (matches OC iso_code_2)
    zone_code: str = ""     # OC zone code, empty when unknown
    telephone: str = ""


@dataclass
class Customer:
    source_id: int
    firstname: str
    lastname: str
    email: str
    telephone: str = ""
    newsletter: bool = False
    created_at: Optional[datetime] = None
    address: Optional[Address] = None
    # Passwords are never migrated. The writer generates random hash+salt.


@dataclass
class OrderAddressSnapshot:
    firstname: str = ""
    lastname: str = ""
    company: str = ""
    address_1: str = ""
    address_2: str = ""
    city: str = ""
    postcode: str = ""
    country_iso: str = ""   # 2-char ISO, may be empty
    country_name: str = ""  # human-readable, snapshot at order time
    zone_name: str = ""     # human-readable, snapshot at order time
    zone_code: str = ""     # OC zone code (resolved by adapter), empty when unknown
    telephone: str = ""


@dataclass
class OrderItem:
    product_source_id: Optional[int]
    product_reference: str   # SKU/model snapshot at order time
    name: str
    quantity: int
    unit_price: float        # tax-exclusive per-unit price as stored by source
    tax_rate: float = 0.0    # percent (e.g. 24.0 for 24%)
    line_total: float = 0.0  # snapshot total for the row (qty * price, etc.)


@dataclass
class Order:
    source_id: int
    reference: str                       # PS order reference (e.g. "XKBKJABCD") — falls back to id
    customer_source_id: Optional[int]
    customer_email: str
    customer_firstname: str
    customer_lastname: str
    payment_method_name: str             # human label, e.g. "Bank wire"
    payment_method_code: str             # module code, e.g. "bankwire"
    currency_code: str                   # ISO 4217, e.g. "EUR"
    currency_value: float = 1.0          # conversion vs. shop default (1.0 if same)
    subtotal: float = 0.0
    shipping_amount: float = 0.0
    tax_amount: float = 0.0
    total: float = 0.0
    status_source: str = ""              # raw source status code (e.g. "5" for PS, "pending" for M1)
    payment_address: Optional[OrderAddressSnapshot] = None
    shipping_address: Optional[OrderAddressSnapshot] = None
    items: list[OrderItem] = field(default_factory=list)
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
