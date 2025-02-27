import os
import subprocess
import json
import re
from transformers import pipeline

def clean_transcript(text):
    text = text.strip()  # Xóa khoảng trắng ở đầu/cuối
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa giữa các từ

    # Tách câu theo dấu chấm (giữ dấu chấm)
    sentences = re.split(r'([.!?])', text)
    
    # Viết hoa chữ cái đầu mỗi câu
    cleaned_sentences = []
    capitalize_next = True  # Đánh dấu cần viết hoa

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if capitalize_next and sentence[0].isalpha():
            sentence = sentence.capitalize()

        cleaned_sentences.append(sentence)
        capitalize_next = sentence in ".!?"  # Viết hoa sau dấu chấm, dấu hỏi, dấu cảm thán

    return " ".join(cleaned_sentences)


def process_audio(input_audio, output_dir="D:/Audio_Training", segment_length=30):
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa có

    # Lấy tên file gốc không có đuôi mở rộng để làm thư mục riêng
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    
    # Tạo thư mục con cho file audio
    file_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(file_output_dir, exist_ok=True)
    # Định dạng file đầu ra (segment)
    split_pattern = os.path.join(file_output_dir, "segment%03d.mp3")  # Định dạng file đầu ra
    subprocess.run([
        "ffmpeg", "-i", input_audio,
        "-f", "segment", "-segment_time", str(segment_length),
        "-c", "copy", split_pattern
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load mô hình Whisper
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")

    transcripts = []
    for file in sorted(os.listdir(file_output_dir)):  # Đọc các file đã tách
        if file.endswith(".mp3") and "segment" in file:
            file_path = os.path.join(file_output_dir, file)
            result = pipe(file_path)  # Nhận diện giọng nói từ âm thanh
            text = result["text"].strip()

            # Làm sạch transcript
            cleaned_text = clean_transcript(text)

            # Lưu transcript vào danh sách
            transcripts.append({
                "audio_filepath": file_path.replace("\\", "/"),  # Định dạng đường dẫn chuẩn
                "text": cleaned_text
            })

    # Lưu transcript vào file JSON, không bị ghi đè bởi file khác
    transcript_file = os.path.join(file_output_dir, f"{base_name}_transcripts.json")
    # Lưu transcript vào file JSON
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=4)

    print(f"Transcript saved at: {transcript_file}")

# Chạy thử với file âm thanh có sẵn
process_audio(r"D:/Audio_Training/Code_Example/Audio_Example/U30_16khz.mp3")
