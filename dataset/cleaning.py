import pandas as pd

# Load the dataset
file_path = "D:\MiniProjectML\dataset\updated_global_foods_hsum.csv"  # เปลี่ยนเป็น path ของไฟล์จริง

df = pd.read_csv(file_path)

# 1. แปลง ingredients จาก string เป็น list

df["ingredients"] = df["ingredients"].apply(lambda x: [i.strip().lower() for i in x.split("+")])

# 2. กำจัดแถวที่ ingredients เป็น "Unknown"
df = df[df["ingredients"].apply(lambda x: "unknown" not in x)]

# 3. ตรวจสอบและลบข้อมูลซ้ำซ้อน
# แปลง ingredients เป็น tuple ชั่วคราวเพื่อลบข้อมูลซ้ำ

df["ingredients"] = df["ingredients"].apply(tuple)
df = df.drop_duplicates()

# ถ้าต้องการลบข้อมูลซ้ำคอลัมน์ en_name
df = df.drop_duplicates(subset=["en_name"], keep="first")

# แปลง ingredients กลับเป็น list

df["ingredients"] = df["ingredients"].apply(list)

# 4. ตรวจสอบค่า null และเติมค่าที่ขาดหาย (ถ้ามี)
df = df.dropna()

# 5. บันทึกไฟล์ที่ถูก clean แล้ว
cleaned_file_path = "D:\MiniProjectML\dataset\cleaned1_thailand_foods.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}")
