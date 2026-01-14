"""
🌿 Few-Shot Plant Disease Classifier
Streamlit Web Application
"""
import streamlit as st
import torch
from PIL import Image
import os
import json
from datetime import datetime
import random

# Import custom modules
from model_loader import load_fewshot_model, get_device, ModelConfig
from few_shot_engine import FewShotClassifier, evaluate_fewshot_accuracy
from utils import (
    load_image, load_images_from_folder, load_support_set_from_folders,
    plot_confusion_matrix, plot_distance_distribution, display_sample_grid,
    create_temp_folder, format_confidence, get_color_by_confidence
)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Few-Shot Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
        color: #1a1a1a !important; /* Đảm bảo chữ luôn rõ ràng */
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZATION ====================
def load_saved_support_set(classifier, base_path="data/diseases"):
    """Tự động quét thư mục và nạp vào classifier"""
    if not os.path.exists(base_path):
        return False, []

    support_data = {}
    all_class_names = []
    
    # Quét các thư mục con
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            display_name = folder_name.replace('_', ' ')
            
            # Load tất cả ảnh trong folder
            tensors = load_images_from_folder(folder_path)
            
            if tensors and len(tensors) > 0:
                # KIỂM TRA VÀ SỬA LỖI TẠI ĐÂY: 
                # Nếu tensors là list của list (ví dụ [[T1, T2]]), hãy làm phẳng nó
                if isinstance(tensors[0], list):
                    tensors = [item for sublist in tensors for item in sublist]
                
                # Đảm bảo mọi phần tử đều là Tensor
                tensors = [t for t in tensors if torch.is_tensor(t)]
                
                if len(tensors) > 0:
                    support_data[display_name] = tensors
                    all_class_names.append(display_name)
    
    if support_data:
        # Gọi build_support_set với dictionary {tên: [tensor1, tensor2, ...]}
        classifier.build_support_set(support_data)
        return True, all_class_names
    
    return False, []

@st.cache_resource
def initialize_model():
    """Load model và khởi tạo classifier"""
    device = get_device()
    model_path = "models/fewshot_encoder_best.pth"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    model, encoder = load_fewshot_model(model_path, device)
    classifier = FewShotClassifier(encoder, device)
    
    return classifier, device

# Khởi tạo
classifier, device = initialize_model()
create_temp_folder('temp')

# --- PHẦN SỬA ĐỔI CHÍNH: TỰ ĐỘNG LOAD KHI START ---
if 'support_set_loaded' not in st.session_state:
    # Thử load dữ liệu đã lưu từ ổ đĩa
    with st.spinner("🚀 Đang tải dữ liệu bệnh đã lưu..."):
        success, loaded_classes = load_saved_support_set(classifier)
        
    if success:
        st.session_state.support_set_loaded = True
        st.session_state.class_names = loaded_classes
    else:
        st.session_state.support_set_loaded = False
        st.session_state.class_names = []

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ==================== HEADER ====================
st.markdown('<div class="main-header">🌿 Few-Shot Plant Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phân loại bệnh cây trồng chỉ với vài mẫu dữ liệu</div>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/2ecc71/ffffff?text=Few-Shot+Learning", use_container_width=True)
    
    st.markdown("## 🎯 Hướng dẫn sử dụng")
    
    mode = st.radio(
        "Chọn chế độ:",
        ["🚀 Quick Test", "📊 Batch Evaluation", "➕ Add New Disease", "📂 Support Set Gallery"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### 📌 Thông tin hệ thống")
    st.info(f"""
    **Device:** {device.upper()}  
    **Model:** EfficientNet-B0  
    **Image Size:** {ModelConfig.IMG_SIZE}x{ModelConfig.IMG_SIZE}
    """)
    
    # Support set info
    if st.session_state.support_set_loaded:
        support_info = classifier.get_support_info()
        st.success(f"""
        ✅ **Support Set Ready**  
        Classes: {support_info['num_classes']}  
        {', '.join(support_info['class_names'][:3])}...
        """)
        
        if st.button("🗑️ Clear Current Session"):
            st.session_state.support_set_loaded = False
            st.session_state.class_names = []
            classifier.prototypes = {}
            classifier.class_names = []
            st.success("✅ Session cleared! (Dữ liệu trên ổ đĩa vẫn còn)")
            st.rerun()
    else:
        st.warning("⚠️ No support set loaded")
    

# ==================== MAIN CONTENT ====================

# ========== MODE 1: QUICK TEST ==========
if mode == "🚀 Quick Test":
    st.markdown("## 🚀 Quick Test Mode")
    st.markdown("Upload **support images** (1-shot) và **query image** để phân loại nhanh")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Support Set")
        st.markdown("Upload 1 ảnh mẫu cho mỗi loại bệnh")
        
        num_classes = st.number_input("Số lượng classes:", min_value=2, max_value=10, value=3)
        
        support_images = {}
        for i in range(num_classes):
            class_name = st.text_input(f"Class {i+1} name:", key=f"class_name_{i}", value=f"Disease_{i+1}")
            uploaded_file = st.file_uploader(f"Upload image for {class_name}", type=['jpg', 'jpeg', 'png'], key=f"support_{i}")
            
            if uploaded_file:
                img_tensor, img_pil = load_image(uploaded_file)
                support_images[class_name] = [img_tensor]
                st.image(img_pil, caption=class_name, width=150)
        
        if st.button("🔨 Build Support Set", key="build_support"):
            if len(support_images) == num_classes:
                with st.spinner("Building support set..."):
                    classifier.build_support_set(support_images)
                    st.session_state.support_set_loaded = True
                    st.session_state.class_names = list(support_images.keys())
                st.success("✅ Support set ready!")
            else:
                st.error("❌ Please upload images for all classes")
    
    with col2:
        st.markdown("### 🔍 Query & Classify")
        
        if not st.session_state.support_set_loaded:
            st.warning("⚠️ Please build support set first")
        else:
            query_file = st.file_uploader("Upload query image", type=['jpg', 'jpeg', 'png'], key="query")
            
            if query_file:
                query_tensor, query_pil = load_image(query_file)
                
                col_img, col_result = st.columns([1, 1])
                
                with col_img:
                    st.image(query_pil, caption="Query Image", use_container_width=True)
                
                with col_result:
                    if st.button("🎯 Classify", key="classify"):
                        with st.spinner("Classifying..."):
                            pred_class, confidence, distances = classifier.classify(query_tensor, return_distances=True)
                        
                        # Display result
                        st.markdown(f"### 🎯 Prediction: **{pred_class}**")
                        st.markdown(f"**Confidence:** {format_confidence(confidence)}")
                        
                        # Distance chart
                        fig = plot_distance_distribution(distances, pred_class)
                        st.pyplot(fig)
                        
                        # Save to history
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'prediction': pred_class,
                            'confidence': confidence
                        })

# ========== MODE 2: BATCH EVALUATION ==========
elif mode == "📊 Batch Evaluation":
    st.markdown("## 📊 Batch Evaluation Mode")
    st.markdown("Đánh giá trên nhiều images cùng lúc")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Load Support Set from Folder")
        
        support_folder = st.text_input("Đường dẫn thư mục support set:", "data/support_set")
        k_shot = st.slider("K-shot (số ảnh mỗi class):", 1, 20, 5)
        
        if st.button("📂 Load Support Set"):
            if os.path.exists(support_folder):
                with st.spinner("Loading support set..."):
                    try:
                        support_imgs, class_names = load_support_set_from_folders(support_folder, k_shot=k_shot)
                        classifier.build_support_set(support_imgs)
                        st.session_state.support_set_loaded = True
                        st.session_state.class_names = class_names
                        
                        st.success(f"✅ Loaded {len(class_names)} classes:")
                        st.write(class_names)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.error(f"❌ Folder not found: {support_folder}")
    
    with col2:
        st.markdown("### 📤 Upload Query Images")
        
        if not st.session_state.support_set_loaded:
            st.warning("⚠️ Please load support set first")
        else:
            query_files = st.file_uploader("Upload multiple query images", 
                                          type=['jpg', 'jpeg', 'png'], 
                                          accept_multiple_files=True)
            
            if query_files and st.button("🎯 Batch Classify"):
                results = []
                query_images = []
                
                progress_bar = st.progress(0)
                
                for i, query_file in enumerate(query_files):
                    query_tensor, query_pil = load_image(query_file)
                    pred_class, confidence = classifier.classify(query_tensor)
                    
                    results.append({
                        'filename': query_file.name,
                        'prediction': pred_class,
                        'confidence': confidence
                    })
                    query_images.append(query_pil)
                    
                    progress_bar.progress((i + 1) / len(query_files))
                
                # Display results table
                st.markdown("### 📋 Results")
                st.dataframe(results, use_container_width=True)
                
                # Display images grid
                st.markdown("### 🖼️ Images Preview")
                titles = [f"{r['prediction']} ({r['confidence']:.2%})" for r in results]
                fig = display_sample_grid(query_images, titles, max_images=9)
                st.pyplot(fig)
                
                # Download results
                import pandas as pd
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False)
                st.download_button("💾 Download Results (CSV)", csv, "results.csv", "text/csv")

# ========== MODE 3: ADD NEW DISEASE ==========
elif mode == "➕ Add New Disease":
    st.markdown("## ➕ Add/Update Disease")
    st.markdown("Thêm loại bệnh mới hoặc bổ sung ảnh mẫu cho bệnh đã có")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Disease Information")
        # Gợi ý các tên bệnh đã có để người dùng dễ chọn nếu muốn cập nhật
        existing_classes = st.session_state.class_names
        st.write(f"Bệnh hiện có: {', '.join(st.session_state.class_names) if st.session_state.class_names else 'Chưa có'}")
        
        new_disease_name = st.text_input("Tên bệnh (Nhập tên mới hoặc tên đã có):", "").strip()
        
        # Kiểm tra xem bệnh đã tồn tại chưa
        is_existing = new_disease_name in existing_classes
        if is_existing:
            st.warning(f"⚠️ Bệnh '{new_disease_name}' đã tồn tại. Ảnh mới sẽ được thêm vào bộ dữ liệu cũ.")
        
        new_disease_desc = st.text_area("Mô tả (optional):", "")
        
        st.markdown("### 📤 Upload Sample Images")
        sample_files = st.file_uploader("Upload thêm ảnh mẫu", 
                                       type=['jpg', 'jpeg', 'png'],
                                       accept_multiple_files=True,
                                       key="new_disease_samples")
        
        if sample_files:
            st.success(f"✅ {len(sample_files)} images prepared")
            
            sample_images = []
            sample_tensors = []
            for f in sample_files:
                tensor, pil_img = load_image(f)
                sample_images.append(pil_img)
                sample_tensors.append(tensor)
            
            st.pyplot(display_sample_grid(sample_images, max_images=6))
            
            btn_label = "🔄 Update Disease" if is_existing else "➕ Add to System"
            
            if st.button(btn_label):
                if new_disease_name:
                    with st.spinner("Processing..."):
                        # 1. Đường dẫn thư mục
                        folder_name = new_disease_name.replace(' ', '_')
                        save_path = f"data/diseases/{folder_name}"
                        os.makedirs(save_path, exist_ok=True)

                        # 2. Đếm số lượng file cũ để đặt tên file mới (tránh ghi đè)
                        existing_files = [f for f in os.listdir(save_path) if f.startswith("sample_")]
                        start_idx = len(existing_files)

                        # 3. Lưu các file ảnh mới
                        for i, f in enumerate(sample_files):
                            file_ext = f.name.split('.')[-1]
                            new_filename = f"sample_{start_idx + i}.{file_ext}"
                            with open(os.path.join(save_path, new_filename), "wb") as img_file:
                                img_file.write(f.getbuffer())

                        # 4. Cập nhật Classifier
                        # Nếu là bệnh cũ, ta nạp lại toàn bộ ảnh (cũ + mới) để tính lại Prototype chính xác nhất
                        all_tensors = load_images_from_folder(save_path)
                        # Đảm bảo all_tensors là list phẳng (xử lý lỗi bạn gặp lúc trước)
                        if all_tensors and isinstance(all_tensors[0], list):
                            all_tensors = [item for sublist in all_tensors for item in sublist]

                        classifier.add_new_class(new_disease_name, all_tensors)
                        
                        # 5. Cập nhật Session State
                        st.session_state.support_set_loaded = True
                        if new_disease_name not in st.session_state.class_names:
                            st.session_state.class_names.append(new_disease_name)
                        
                        # 6. Lưu/Cập nhật metadata
                        metadata = {
                            'name': new_disease_name,
                            'description': new_disease_desc,
                            'total_samples': len(all_tensors),
                            'last_updated': datetime.now().isoformat()
                        }
                        with open(os.path.join(save_path, "metadata.json"), 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                            
                    st.success(f"✅ Đã cập nhật thành công bệnh '{new_disease_name}'! Tổng số ảnh hiện tại: {len(all_tensors)}")
                else:
                    st.error("❌ Vui lòng nhập tên bệnh")
# ========== MODE 4: SUPPORT SET GALLERY ==========
elif mode == "📂 Support Set Gallery":
    st.markdown("## 📂 Support Set Gallery")
    
    if not st.session_state.class_names:
        st.warning("⚠️ Hiện chưa có loại bệnh nào được nạp vào hệ thống.")
    else:
        selected_disease = st.selectbox("Chọn loại bệnh:", st.session_state.class_names)

        if selected_disease:
            folder_name = selected_disease.replace(' ', '_')
            base_path = f"data/diseases/{folder_name}"
            metadata_path = os.path.join(base_path, "metadata.json")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📋 Thông tin")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    
                    # Giải pháp dùng Markdown chuẩn Streamlit - Chắc chắn không lỗi màu
                    st.success(f"**Tên bệnh:** {meta.get('name', selected_disease)}")
                    st.info(f"**Mô tả:** {meta.get('description', 'Không có mô tả')}")
                    
                    # Hiển thị số liệu dạng metric cho đẹp
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Số mẫu", f"{meta.get('total_samples', 'N/A')}")
                    m_col2.metric("Cập nhật", f"{meta.get('last_updated', 'N/A')[:10]}")
                else:
                    st.warning("Bệnh này không có file metadata.json")

            with col2:
                st.markdown("### 🖼️ Ảnh mẫu ngẫu nhiên (Random 3)")
                sample_imgs = []
                if os.path.exists(base_path):
                    valid_extensions = ('.png', '.jpg', '.jpeg')
                    files = [f for f in os.listdir(base_path) if f.lower().endswith(valid_extensions)]
                    
                    # LẤY 3 ẢNH NGẪU NHIÊN
                    if len(files) > 0:
                        num_to_sample = min(3, len(files))
                        random_files = random.sample(files, num_to_sample)
                        
                        for f_name in random_files:
                            img_path = os.path.join(base_path, f_name)
                            sample_imgs.append(Image.open(img_path))

                if sample_imgs:
                    # Hiển thị 3 ảnh trên 3 cột
                    img_cols = st.columns(3)
                    for idx, img in enumerate(sample_imgs):
                        img_cols[idx].image(img, use_container_width=True, caption=f"Sample {idx+1}")
                else:
                    st.error("Không tìm thấy ảnh mẫu trong thư mục.")
# ==================== FOOTER ====================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Statistics")
    if st.session_state.prediction_history:
        st.metric("Total Predictions", len(st.session_state.prediction_history))
    else:
        st.metric("Total Predictions", 0)

with col2:
    st.markdown("### 🎯 Classes")
    st.metric("Loaded Classes", len(st.session_state.class_names))

with col3:
    st.markdown("### 🔋 System")
    st.metric("Device", device.upper())

# Prediction history
if st.session_state.prediction_history:
    with st.expander("📜 View Prediction History"):
        for i, pred in enumerate(reversed(st.session_state.prediction_history[-10:])):
            st.text(f"{pred['timestamp']} | {pred['prediction']} | {pred['confidence']:.2%}")