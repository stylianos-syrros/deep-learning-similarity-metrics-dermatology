# IoU scores από το αποτέλεσμα σου
iou_scores = [0.2759666881613191, 0.21176429166711275, 0.63832270589847, 0.15243241825264622, 0.0, 0.0]
#iou_scores = [0.5202885841836735, 0.9947285554846939, 0.9722576530612246, 0.5251924815658422, 0.1875, 0.0]

# Υπολογισμός του αθροίσματος των IoU scores
iou_sum = sum(iou_scores)

# Αν δεν θέλουμε να λάβουμε υπόψη τα επίπεδα με IoU = 0
iou_scores = [score for score in iou_scores if score > 0]
iou_sum = sum(iou_scores)

# Κανονικοποίηση για να βρούμε τα βάρη
weights = [score / iou_sum for score in iou_scores]

# Εκτύπωση των βαρών
for i, w in enumerate(weights):
    print(f"Weight for Layer {i+1}: {w}")
