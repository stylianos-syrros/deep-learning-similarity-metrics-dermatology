from PIL import Image

# Φορτώνει τη φωτογραφία σε μορφή PIL
image_path = r"C:\Users\steli\DIPLOMA\bcc\CASE139\1.jpeg"
image = Image.open(image_path)

# Κρατά την αρχική μορφή της φωτογραφίας χωρίς αλλαγές
image = image.convert("RGB")

# Αποθηκεύει τη φωτογραφία σε μορφή JPEG
output_path1 = r"C:\Users\steli\DIPLOMA\bcc\CASE139\1.jpg"
output_path2 = r"C:\Users\steli\DIPLOMA\bcc\CASE139\2.jpg"
image.save(output_path1)
image.save(output_path2)

