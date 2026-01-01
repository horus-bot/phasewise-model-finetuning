import os

folder = r"C:\Users\lenovo\Desktop\image classicification\Imagenes"

for file in os.listdir(folder):
    if "test" in file.lower():
        os.remove(os.path.join(folder, file))
        print("❌ Deleted:", file)

print(" Cleanup complete. All *_test.jpg removed.")
#check
print("hy")