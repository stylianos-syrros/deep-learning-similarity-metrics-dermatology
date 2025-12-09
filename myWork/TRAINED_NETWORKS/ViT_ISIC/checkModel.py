import torch

# Φόρτωση του checkpoint
model_path = r"D:\Diploma\ViT_models_ISIC\best_vit_model.pth"
checkpoint = torch.load(model_path)

# Εμφάνιση των κλειδιών στο state_dict του checkpoint
print("Keys in checkpoint['model_state_dict']: ", checkpoint['model_state_dict'].keys())

# Δημιουργία του μοντέλου
import timm
trained_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)

# Εμφάνιση των κλειδιών στο state_dict του μοντέλου
model_state_dict = trained_model.state_dict()
print("Keys in trained_model.state_dict(): ", model_state_dict.keys())

# Επιλογή της σωστής προσέγγισης με βάση την ανάλυση
if 'head.weight' in checkpoint['model_state_dict']:
    print("Using strict loading.")
    trained_model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Removing head layers and using non-strict loading.")
    checkpoint_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith('head.')}
    trained_model.load_state_dict(checkpoint_state_dict, strict=False)
