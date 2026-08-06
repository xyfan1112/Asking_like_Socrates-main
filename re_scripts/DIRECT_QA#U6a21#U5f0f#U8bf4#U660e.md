# Direct QA 模式说明

正式 B0/B1 Direct 消融默认使用：

```bash
DIRECT_QA_MODE=legacy_class_roi_obb
```

探索格式：

```bash
DIRECT_QA_MODE=all_image_json_obb
DIRECT_QA_MODE=all_image_json_hbb
DIRECT_QA_MODE=single_ref_json_obb
```

其中 `all_image_json_hbb` 精确对应 `bbox_2d + categories` 平行数组，但会丢失 OBB 旋转信息。更稳的全图 JSON 是：

```json
{
  "objects": [
    {
      "category": "classes.txt中的标准类别",
      "obb_8": [x1, y1, x2, y2, x3, y3, x4, y4]
    }
  ]
}
```

非 legacy 模式训练后继续运行原 test-matrix 属于跨协议迁移评测，需要：

```bash
ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX=1
```
