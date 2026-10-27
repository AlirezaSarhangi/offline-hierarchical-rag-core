"""
رجیستری لایه و گره سازمانی (`LayerNodeRegistry`) طبق بخش ۶.۱، بخش ۶.۲ و جدول ۱۹.۱ پروپوزال.

این ماژول وظیفه ثبت و اعتبارسنجی سلسله‌مراتب ۵ لایه‌ای (لایه ۰ تا لایه ۴)، عضویت یکتای هر گره
در دقیقاً یک لایه اصلی معتبر، اعتبارسنجی روابط والد و فرزند میان لایه‌ها، مدیریت مرز گره نهایی
(`is_leaf_boundary`)، ثبت قواعد گزارش‌دهی و انتقال دانش به لایه بالاتر، و سریال‌سازی/بازسازی
در قالب دیکشنری و JSON را بر عهده دارد.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.domain.entities import Layer, Node
from src.domain.exceptions import (
    DomainValidationError,
    InvalidLayerHierarchyError,
    LeafNodeBoundaryViolationError,
    RegistryNotFoundError,
)
from src.domain.value_objects import (
    LayerIndex,
    NodeIdentifier,
    TransferDetailLevel,
)


DEFAULT_LAYER_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {
        "layer_index": 0,
        "name": "لایه صفر: لایه راهبردی",
        "role_description": "مرجع عالی تصمیم‌گیری یا مالک کلان ساختار سازمانی در سناریوی نمونه",
        "trackable_info_summary": (
            "دریافت خلاصه راهبردی دانش، مسائل کلیدی حل‌نشده، وضعیت RFP، شکاف‌های پژوهشی و اولویت‌های بین‌گرهی"
        ),
        "default_detail_level": TransferDetailLevel.EXECUTIVE_HIGH,
    },
    {
        "layer_index": 1,
        "name": "لایه یک: سازمان‌های مادر",
        "role_description": "شرکت‌های اصلی، سازمان‌های اصلی، ستادها یا معاونت‌های راهبردی",
        "trackable_info_summary": (
            "دریافت خلاصه مدیریتی، ریسک‌ها و دستاوردهای زیرمجموعه‌ها"
        ),
        "default_detail_level": TransferDetailLevel.MANAGERIAL,
    },
    {
        "layer_index": 2,
        "name": "لایه دو: زیرمجموعه‌ها و واحدهای تخصصی",
        "role_description": "شرکت‌های تابعه، مراکز پژوهشی، واحدهای مهندسی، واحدهای عملیاتی یا دفاتر برنامه",
        "trackable_info_summary": (
            "نگهداری دانش تفصیلی و آماده‌سازی گزارش‌های کنترل‌شده به لایه بالاتر"
        ),
        "default_detail_level": TransferDetailLevel.TECHNICAL_SUMMARY,
    },
    {
        "layer_index": 3,
        "name": "لایه سه: پروژه‌ها، تیم‌ها و گره‌های اجرایی",
        "role_description": "تیم‌های پروژه، واحدهای پیاده‌سازی، گروه‌های پژوهشی، تیم‌های میدانی و گروه‌های مجری",
        "trackable_info_summary": (
            "نگهداری جزئی‌ترین دانش فنی، عملیاتی، نیروی انسانی و مسئله‌محور"
        ),
        "default_detail_level": TransferDetailLevel.TECHNICAL_DETAILED,
    },
    {
        "layer_index": 4,
        "name": "لایه چهار: منابع داخلی دانش اختیاری",
        "role_description": "کتابخانه‌ها، مجموعه اسناد، سوابق پرسنلی، آرشیو پروژه یا مخازن محلی",
        "trackable_info_summary": (
            "در صورت نیاز مطالعه موردی، به‌عنوان منبع دانش مستقل مدل می‌شوند"
        ),
        "default_detail_level": TransferDetailLevel.TECHNICAL_DETAILED,
    },
)


class LayerNodeRegistry:
    """
    رجیستری مرکزی لایه‌ها و گره‌های سازمانی (Aggregate Root برای زمینه سلسله‌مراتب سازمانی).

    ناوردایی‌های کلیدی (Invariants):
    1. هر لایه دارای شماره یکتا بین ۰ تا ۴ است.
    2. هر گره دقیقاً به یک لایه اصلی معتبر تعلق دارد و نمی‌تواند در بیش از یک لایه عضو باشد.
    3. گره‌های لایه ۰ (لایه راهبردی) فاقد گره والد هستند.
    4. گره‌های لایه‌های ۱ تا ۴ باید دارای گره والد ثبت‌شده در رجیستری باشند و لایه گره والد
       باید اکیداً بالاتر از لایه گره فرزند باشد (`parent.layer_index.value < child.layer_index.value`).
    5. اگر گره‌ای دارای ویژگی `is_leaf_boundary=True` باشد (مرز گره نهایی)، ثبت هرگونه گره فرزند
       برای آن ممنوع است و استثنای `LeafNodeBoundaryViolationError` پرتاب می‌شود.
    """

    def __init__(self, initialize_standard_layers: bool = True) -> None:
        self._layers: Dict[int, Layer] = {}
        self._nodes: Dict[str, Node] = {}
        if initialize_standard_layers:
            self.initialize_default_layers()

    def initialize_default_layers(self) -> None:
        """ثبت ۵ لایه استاندارد پیشنهادی پروپوزال (جدول ۶.۲: لایه‌های ۰ تا ۴)."""
        for item in DEFAULT_LAYER_DEFINITIONS:
            layer_idx = LayerIndex(item["layer_index"])
            if layer_idx.value not in self._layers:
                self._layers[layer_idx.value] = Layer(
                    layer_index=layer_idx,
                    name=item["name"],
                    role_description=item["role_description"],
                    trackable_info_summary=item["trackable_info_summary"],
                    default_detail_level=item["default_detail_level"],
                )

    def register_layer(self, layer: Layer, overwrite: bool = False) -> Layer:
        """ثبت یا به‌روزرسانی یک لایه سازمانی در رجیستری."""
        if not isinstance(layer, Layer):
            raise DomainValidationError(
                "ورودی متد register_layer باید از نوع موجودیت Layer باشد.",
                field_name="layer",
            )
        idx = layer.layer_index.value
        if idx in self._layers and not overwrite:
            existing_members = self._layers[idx].member_node_ids
            for member_id in existing_members:
                layer.add_member_node(member_id)
        self._layers[idx] = layer
        return layer

    def register_node(self, node: Node) -> Node:
        """
        ثبت گره سازمانی جدید در رجیستری با اعمال کامل قواعد سلسله‌مراتب، یکتایی لایه و مرز گره نهایی.
        """
        if not isinstance(node, Node):
            raise DomainValidationError(
                "ورودی متد register_node باید از نوع موجودیت Node باشد.",
                field_name="node",
            )

        node_key = node.node_id.value
        target_layer_num = node.layer_index.value

        # ۱. بررسی وجود لایه هدف در رجیستری
        if target_layer_num not in self._layers:
            raise RegistryNotFoundError(
                entity_type="Layer", identifier=str(target_layer_num)
            )

        # ۲. بررسی عدم ثبت تکراری گره یا تغییر غیرمجاز لایه اصلی
        if node_key in self._nodes:
            existing_node = self._nodes[node_key]
            if existing_node.layer_index != node.layer_index:
                raise InvalidLayerHierarchyError(
                    f"گره '{node_key}' قبلاً در لایه {existing_node.layer_index.value} ثبت شده است و "
                    f"نمی‌تواند هم‌زمان به لایه {target_layer_num} تعلق داشته باشد (هر گره دقیقاً یک لایه اصلی دارد)."
                )
            raise DomainValidationError(
                f"گره با شناسه '{node_key}' قبلاً در رجیستری ثبت شده است.",
                field_name="node_id",
            )

        # ۳. بررسی اینکه گره در هیچ لایه دیگری عضو نباشد
        for l_idx, layer_obj in self._layers.items():
            if l_idx != target_layer_num and layer_obj.has_member(node.node_id):
                raise InvalidLayerHierarchyError(
                    f"نقض یکتایی لایه: گره '{node_key}' در لایه {l_idx} عضویت دارد و نمی‌تواند در لایه {target_layer_num} ثبت شود."
                )

        # ۴. اعتبارسنجی رابطه والد و فرزند و قانون مرز گره نهایی
        if node.layer_index.is_strategic_root:
            if node.parent_node_id is not None:
                raise InvalidLayerHierarchyError(
                    f"گره لایه راهبردی ('{node_key}') در لایه ۰ نمی‌تواند دارای گره والد باشد."
                )
        else:
            if node.parent_node_id is None:
                raise InvalidLayerHierarchyError(
                    f"گره '{node_key}' در لایه {target_layer_num} باید دارای گره والد معتبر باشد."
                )
            parent_key = node.parent_node_id.value
            if parent_key not in self._nodes:
                raise RegistryNotFoundError(
                    entity_type="ParentNode", identifier=parent_key
                )
            parent_node = self._nodes[parent_key]

            # بررسی بالاتر بودن لایه والد نسبت به لایه فرزند
            if not parent_node.layer_index.is_higher_authority_than(node.layer_index):
                raise InvalidLayerHierarchyError(
                    f"سلسله‌مراتب لایه نامعتبر: گره والد '{parent_key}' در لایه {parent_node.layer_index.value} قرار دارد "
                    f"که بالاتر از لایه گره فرزند '{node_key}' (لایه {target_layer_num}) نیست."
                )

            # بررسی مرز گره نهایی (is_leaf_boundary) روی والد
            if parent_node.is_leaf_boundary:
                raise LeafNodeBoundaryViolationError(
                    parent_node_id=parent_key,
                    child_node_id=node_key,
                )

            # ثبت فرزند در لیست فرزندان گره والد
            parent_node.register_child(node.node_id)

        # ۵. ثبت نهایی گره و عضویت در لایه مربوطه
        self._nodes[node_key] = node
        self._layers[target_layer_num].add_member_node(node.node_id)
        return node

    def get_layer(self, layer_index: Union[int, LayerIndex]) -> Layer:
        """بازیابی یک لایه بر اساس شماره یا شیء `LayerIndex`."""
        idx = layer_index.value if isinstance(layer_index, LayerIndex) else LayerIndex(layer_index).value
        if idx not in self._layers:
            raise RegistryNotFoundError(entity_type="Layer", identifier=str(idx))
        return self._layers[idx]

    def get_node(self, node_id: Union[str, NodeIdentifier]) -> Node:
        """بازیابی یک گره ثبت‌شده بر اساس شناسه."""
        key = node_id.value if isinstance(node_id, NodeIdentifier) else NodeIdentifier(node_id).value
        if key not in self._nodes:
            raise RegistryNotFoundError(entity_type="Node", identifier=key)
        return self._nodes[key]

    def get_children(self, node_id: Union[str, NodeIdentifier]) -> List[Node]:
        """دریافت فهرست موجودیت‌های گره فرزند مستقیم برای یک گره."""
        parent = self.get_node(node_id)
        return [self._nodes[cid.value] for cid in parent.child_node_ids if cid.value in self._nodes]

    def get_nodes_by_layer(self, layer_index: Union[int, LayerIndex]) -> List[Node]:
        """دریافت تمامی گره‌های عضو یک لایه مشخص."""
        layer = self.get_layer(layer_index)
        return [self._nodes[mid.value] for mid in layer.member_node_ids if mid.value in self._nodes]

    def get_ancestry_chain(self, node_id: Union[str, NodeIdentifier]) -> List[Node]:
        """
        دریافت زنجیره بالادستی (از خود گره تا گره ریشه در لایه ۰) جهت مسیریابی انتقال دانش.
        """
        current = self.get_node(node_id)
        chain: List[Node] = [current]
        visited = {current.node_id.value}
        while current.parent_node_id is not None:
            parent_key = current.parent_node_id.value
            if parent_key in visited:
                raise InvalidLayerHierarchyError(
                    f"حلقه معیوب در سلسله‌مراتب والد-فرزند برای گره '{parent_key}' تشخیص داده شد."
                )
            visited.add(parent_key)
            current = self.get_node(parent_key)
            chain.append(current)
        return chain

    def update_reporting_rules(
        self,
        node_id: Union[str, NodeIdentifier],
        reporting_rules: Dict[str, Any],
        required_detail_level: Optional[TransferDetailLevel] = None,
    ) -> Node:
        """به‌روزرسانی قواعد گزارش‌دهی و سطح جزئیات انتقال دانش یک گره به لایه بالاتر."""
        node = self.get_node(node_id)
        if not isinstance(reporting_rules, dict):
            raise DomainValidationError(
                "قواعد گزارش‌دهی (reporting_rules) باید یک دیکشنری معتبر باشد.",
                field_name="reporting_rules",
            )
        node.reporting_rules.update(reporting_rules)
        if required_detail_level is not None:
            node.required_transfer_detail_level = required_detail_level
        return node

    def set_leaf_boundary(
        self, node_id: Union[str, NodeIdentifier], is_leaf_boundary: bool = True
    ) -> Node:
        """
        تنظیم وضعیت مرز گره نهایی (`is_leaf_boundary`).
        در صورتی که گره دارای زیرمجموعه ثبت‌شده باشد، علامت‌گذاری آن به عنوان گره نهایی ممنوع است.
        """
        node = self.get_node(node_id)
        if is_leaf_boundary:
            node.mark_as_leaf_boundary()
        else:
            node.is_leaf_boundary = False
        return node

    def validate_integrity(self) -> bool:
        """
        بررسی جامع یکپارچگی رجیستری:
        - تعلق دقیق هر گره به یک لایه اصلی معتبر
        - صحت روابط والد و فرزند در تمام گره‌ها
        - عدم نقض مرز گره نهایی
        """
        membership_count: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for l_idx, layer in self._layers.items():
            for member_id in layer.member_node_ids:
                if member_id.value not in self._nodes:
                    raise RegistryNotFoundError(
                        entity_type="Node", identifier=member_id.value
                    )
                node_obj = self._nodes[member_id.value]
                if node_obj.layer_index.value != l_idx:
                    raise InvalidLayerHierarchyError(
                        f"عدم تطابق لایه برای گره '{member_id.value}': ثبت در لایه {l_idx} ولی دارای شاخص {node_obj.layer_index.value}."
                    )
                membership_count[member_id.value] += 1

        for nid, count in membership_count.items():
            if count != 1:
                raise InvalidLayerHierarchyError(
                    f"گره '{nid}' باید دقیقاً در ۱ لایه عضو باشد؛ تعداد عضویت فعلی: {count}."
                )

        for nid, node in self._nodes.items():
            if node.is_leaf_boundary and node.child_node_ids:
                raise LeafNodeBoundaryViolationError(
                    parent_node_id=nid,
                    child_node_id=node.child_node_ids[0].value,
                )
            if not node.layer_index.is_strategic_root:
                if node.parent_node_id is None:
                    raise InvalidLayerHierarchyError(
                        f"گره غیرراهبردی '{nid}' فاقد والد است."
                    )
                parent = self.get_node(node.parent_node_id)
                if not parent.layer_index.is_higher_authority_than(node.layer_index):
                    raise InvalidLayerHierarchyError(
                        f"لایه والد '{parent.node_id.value}' بالاتر از لایه فرزند '{nid}' نیست."
                    )

        return True

    @property
    def layers(self) -> List[Layer]:
        """فهرست مرتب‌شده تمامی لایه‌های ثبت‌شده از ۰ تا ۴."""
        return [self._layers[k] for k in sorted(self._layers.keys())]

    @property
    def nodes(self) -> List[Node]:
        """فهرست تمامی گره‌های ثبت‌شده در رجیستری."""
        return list(self._nodes.values())

    def to_dict(self) -> Dict[str, Any]:
        """خروجی دیکشنری کامل رجیستری لایه‌ها و گره‌ها."""
        return {
            "schema_version": "1.0.0",
            "total_layers": len(self._layers),
            "total_nodes": len(self._nodes),
            "layers": [layer.to_dict() for layer in self.layers],
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LayerNodeRegistry:
        """بازسازی کامل `LayerNodeRegistry` از دیکشنری و اعتبارسنجی مجدد تمام قواعد."""
        if not isinstance(data, dict) or "layers" not in data or "nodes" not in data:
            raise DomainValidationError(
                "دیکشنری ورودی برای بازسازی LayerNodeRegistry باید شامل کلیدهای 'layers' و 'nodes' باشد.",
                field_name="data",
            )
        registry = cls(initialize_standard_layers=False)
        for layer_dict in data["layers"]:
            layer_obj = Layer(
                layer_index=LayerIndex(layer_dict["layer_index"]),
                name=layer_dict["name"],
                role_description=layer_dict["role_description"],
                trackable_info_summary=layer_dict["trackable_info_summary"],
                default_detail_level=TransferDetailLevel(
                    layer_dict["default_detail_level"]
                ),
                member_node_ids=[],
            )
            registry.register_layer(layer_obj, overwrite=True)

        # مرتب‌سازی گره‌ها بر اساس شماره لایه تا همیشه والد پیش از فرزند ثبت گردد
        sorted_nodes_data = sorted(
            data["nodes"], key=lambda item: item["layer_index"]
        )
        for node_dict in sorted_nodes_data:
            node_obj = Node(
                node_id=NodeIdentifier(node_dict["node_id"]),
                name=node_dict["name"],
                layer_index=LayerIndex(node_dict["layer_index"]),
                domain_of_activity=node_dict["domain_of_activity"],
                parent_node_id=(
                    NodeIdentifier(node_dict["parent_node_id"])
                    if node_dict.get("parent_node_id")
                    else None
                ),
                is_leaf_boundary=bool(node_dict.get("is_leaf_boundary", False)),
                reporting_rules=dict(node_dict.get("reporting_rules", {})),
                required_transfer_detail_level=TransferDetailLevel(
                    node_dict.get(
                        "required_transfer_detail_level",
                        TransferDetailLevel.MANAGERIAL.value,
                    )
                ),
                confidentiality_constraints=tuple(
                    node_dict.get("confidentiality_constraints", ["INTERNAL"])
                ),
                child_node_ids=[],
            )
            registry.register_node(node_obj)

        registry.validate_integrity()
        return registry

    def to_json(self, indent: int = 2) -> str:
        """تبدیل رجیستری به رشته JSON با پشتیبانی کامل از کاراکترهای فارسی."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> LayerNodeRegistry:
        """بازسازی رجیستری از رشته JSON."""
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise DomainValidationError(
                f"رشته JSON ورودی رجیستری معتبر نیست: {exc}",
                field_name="json_str",
            ) from exc
        return cls.from_dict(parsed)

    def save_to_json_file(self, file_path: Union[str, Path]) -> Path:
        """ذخیره رجیستری در فایل JSON با کدگذاری UTF-8."""
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(self.to_json(indent=2), encoding="utf-8")
        return target_path

    @classmethod
    def load_from_json_file(cls, file_path: Union[str, Path]) -> LayerNodeRegistry:
        """بارگذاری و اعتبارسنجی رجیستری از فایل JSON."""
        target_path = Path(file_path)
        if not target_path.exists():
            raise RegistryNotFoundError(
                entity_type="RegistryFile", identifier=str(target_path)
            )
        content = target_path.read_text(encoding="utf-8")
        return cls.from_json(content)


__all__ = ["DEFAULT_LAYER_DEFINITIONS", "LayerNodeRegistry"]
