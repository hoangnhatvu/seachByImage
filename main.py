from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import subprocess
import shutil
import os
import tempfile

app = FastAPI()

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    # Tạo một tệp tạm thời với phần mở rộng giống với tệp tải lên
    suffix = os.path.splitext(image.filename)[1]  # Lấy phần mở rộng của tệp tải lên
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_image:
        temp_image_path = temp_image.name
        # Lưu ảnh tải lên vào tệp tạm thời
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    
    # Đường dẫn tới mô hình và lệnh YOLO
    model_path = "./best.pt"
    yolo_command = [
        "yolo",
        "task=classify",
        "mode=predict",
        f"model={model_path}",
        "conf=0.25",
        f"source={temp_image_path}"
    ]
    
    # Chạy lệnh YOLO và thu thập kết quả
    try:
        result = subprocess.run(yolo_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        
        # Trích xuất thông tin nhãn và xác suất từ kết quả YOLO
        output = result.stdout
        lines = output.strip().split('\n')
        
        # Tìm dòng chứa nhãn và xác suất
        for line in lines:
            if "image" in line and ".jpg" or ".png" in line:
                label_prob_line = line
                break
        
        # Trích xuất nhãn và xác suất từ dòng
        label_probs = label_prob_line.split('128x128 ')[1].split(', ')
        result_str = (label_probs[0].split(' '))[0].replace("_", " ")
        
        # Xóa ảnh tạm thời sau khi xử lý xong
        os.remove(temp_image_path)
        
        return JSONResponse(content={"result": result_str})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
