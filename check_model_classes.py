#!/usr/bin/env python3
"""Kiểm tra xem custom trained model có pig class không."""

import torch

# Load model bằng YOLOv5 format
model = torch.hub.load(".", "custom", path="runs/train/exp3/weights/best.pt", source="local", force_reload=True)

print("=" * 60)
print("🔍 KIỂM TRA CUSTOM TRAINED MODEL")
print("=" * 60)
print()
print("Model classes:")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")

print()
print("Tổng số classes:", len(model.names))
print()

if "pig" in model.names.values():
    print("✅ 'pig' ĐÃ CÓ TRONG MODEL")
else:
    print("❌ 'pig' CHƯA CÓ TRONG MODEL")

print()
print("Confidence threshold hiện tại:", model.conf)
print()
print("=" * 60)
