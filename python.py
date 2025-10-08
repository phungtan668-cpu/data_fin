import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import time # Dùng cho việc mô phỏng độ trễ/backoff nếu cần

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# ************************* KHỞI TẠO SESSION STATE VÀ CHAT HISTORY *************************
if "chat_history" not in st.session_state:
    # Lịch sử chat: [{"role": "user", "parts": [{"text": "..."}]}, ...]
    st.session_state["chat_history"] = []
if "data_context" not in st.session_state:
    # Dữ liệu tài chính đã xử lý (dưới dạng markdown string) để cung cấp context cho AI
    st.session_state["data_context"] = ""
# ******************************************************************************************


# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""

    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]

    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 thủ công cho mẫu số để tính tỷ trọng
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100

    return df

# --- Hàm gọi API Gemini cho Nhận xét Tự động (Chức năng 5) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét tự động."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.

        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Hàm xử lý khung chat tương tác (Chức năng 6) ---
def handle_chat_query(user_prompt, api_key, data_context):
    """Xử lý yêu cầu chat, duy trì ngữ cảnh dữ liệu và lịch sử chat."""
    
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Lỗi khởi tạo Gemini Client: {e}")
        return "Không thể khởi tạo Gemini Client."

    model_name = 'gemini-2.5-flash'

    # Xây dựng System Instruction (Hướng dẫn cho AI)
    system_instruction = f"""
    Bạn là một chuyên gia phân tích tài chính thông minh và nhiệt tình.
    Ngữ cảnh Dữ liệu đã xử lý của doanh nghiệp (bao gồm Chỉ tiêu, Năm trước, Năm sau, Tốc độ tăng trưởng và Tỷ trọng cơ cấu) được cung cấp trong dấu ngoặc kép 3 lần:
    \"\"\"
    {data_context}
    \"\"\"
    Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng một cách chính xác, dựa trên dữ liệu bạn được cung cấp.
    Bạn PHẢI tham khảo dữ liệu này khi trả lời. Nếu dữ liệu không đủ để trả lời, hãy nói rõ điều đó.
    """
    
    # 1. Chuẩn bị contents (bao gồm System Instruction và lịch sử chat)
    
    # Thêm System Instruction làm hướng dẫn đầu tiên (sử dụng role 'user' để thiết lập ngữ cảnh ẩn)
    contents = [
        {"role": "user", "parts": [{"text": system_instruction}]}
    ]
    
    # Thêm toàn bộ lịch sử chat hiện tại (lưu ý: lịch sử chat trong session_state CHỈ chứa user/model xen kẽ)
    contents.extend(st.session_state["chat_history"])
    
    # Thêm prompt hiện tại của user vào contents
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})
    
    # Cập nhật lịch sử chat hiển thị (chỉ thêm tin nhắn user mới nhất)
    st.session_state["chat_history"].append({"role": "user", "parts": [{"text": user_prompt}]})


    # 2. Gửi yêu cầu tới Gemini
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )
        ai_response = response.text
        
        # 3. Cập nhật lịch sử chat với câu trả lời của AI
        st.session_state["chat_history"].append({"role": "model", "parts": [{"text": ai_response}]})
        return ai_response

    except APIError as e:
        return f"Lỗi gọi Gemini API (Chat): Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong chat: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)

        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']

        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            # ******************************* CẬP NHẬT DATA CONTEXT CHO CHAT *******************************
            st.session_state["data_context"] = df_processed.to_markdown(index=False)
            # **********************************************************************************************

            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)

            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")

            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )

            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                thanh_toan_hien_hanh_N_1 = "N/A"

            # --- Chức năng 5: Nhận xét AI (Tự động) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")

            # Chuẩn bị dữ liệu để gửi cho AI (Tương tự code gốc)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)',
                    'Tăng trưởng Tài sản ngắn hạn (%)',
                    'Thanh toán hiện hành (N-1)',
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    (f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else 'N/A'),
                    f"{thanh_toan_hien_hanh_N_1}",
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False)

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY")

                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# ************************* CHỨC NĂNG MỚI: KHUNG CHAT HỎI ĐÁP *************************

st.markdown("---") # Đường kẻ ngang để phân biệt
st.subheader("6. Chat Hỏi đáp Chuyên sâu (Gemini)")

if st.session_state["data_context"]:
    # 1. Hiển thị Lịch sử Chat
    for message in st.session_state["chat_history"]:
        # Đảm bảo chỉ hiển thị tin nhắn có text (tránh hiển thị system instruction ẩn)
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["parts"][0]["text"])

    # 2. Xử lý Input từ người dùng
    user_prompt = st.chat_input("Hỏi Gemini bất kỳ câu hỏi nào về Báo cáo Tài chính đã tải lên (ví dụ: 'Tài sản ngắn hạn tăng trưởng bao nhiêu phần trăm?').")

    if user_prompt:
        # Lấy API Key
        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Không thể bắt đầu Chat.")
        else:
            # Hiển thị prompt của user ngay lập tức
            # Việc này đã được thực hiện trong hàm handle_chat_query, nên ta chỉ cần gọi hàm
            
            with st.spinner("Gemini đang phân tích và trả lời..."):
                # Gọi hàm chat, hàm này sẽ tự động cập nhật session_state và hiển thị câu trả lời
                handle_chat_query(user_prompt, api_key, st.session_state["data_context"])
                # Streamlit sẽ tự động rerun và hiển thị tin nhắn mới nhất
            st.rerun()
    
else:
    st.warning("Vui lòng tải lên và xử lý dữ liệu tài chính (Bước 1) trước khi bắt đầu khung chat.")

# *************************************************************************************
