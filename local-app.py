import os
import io
import time
import json
import base64
import tempfile
import concurrent.futures

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

#############################
# Gemini / Local LLM Helper
#############################
def get_answer_and_best_chunks(user_query, evidence):
    """
    Sends user_query and evidence to Gemini.
    Returns a dict with "answer", "reasoning", and "best_chunks".
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not set in .env! Cannot process question.")
        return {
            "answer": "Error: GEMINI_API_KEY is missing.",
            "reasoning": "Missing API key in environment variables.",
            "best_chunks": []
        }
        
    prompt = f"""
    Use the following JSON evidence extracted from the uploaded PDF files, answer the following question based on that evidence.
    Please return ONLY your response in JSON format with exactly three keys: 
    1. "answer": Your detailed answer to the question
    2. "reasoning": Your step-by-step reasoning process
    3. "best_chunks": A list of objects with:
       - "file"
       - "page"
       - "bboxes" (each bbox is [x, y, w, h])
       - "captions" (list of text snippets)
       - "reason"
       
    Question: {user_query}

    Evidence: {evidence}
    """

    try:
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        
        raw = response.text.strip()
        
        # Robust JSON extraction using regex
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
            
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        import traceback
        with open("gemini_error.log", "w", encoding="utf-8") as f:
            f.write(f"Exception: {str(e)}\n\n")
            f.write(traceback.format_exc() + "\n\n")
            try:
                f.write(f"Raw Response object: {response}\n")
                if hasattr(response, 'text'):
                    f.write(f"Response Text: {response.text}\n")
            except Exception as inner_e:
                f.write(f"Could not read response text: {inner_e}\n")
                
        st.error(f"Error parsing answer from Gemini. Check gemini_error.log for details. Error: {e}")
        return {
            "answer": "Sorry, I could not retrieve an answer.",
            "reasoning": f"An error occurred during Gemini processing: {e}",
            "best_chunks": []
        }

#############################
# Extract page dimensions for a PDF
#############################
def get_page_dims(pdf_file):
    """
    Extracts the width and height dimensions for each page in a given PDF file.
    
    Args:
        pdf_file: A Streamlit UploadedFile object containing the PDF data.
        
    Returns:
        list of tuples: A list where each element is a (width, height) tuple corresponding to a page.
    """
    page_dims = []
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in pdf_document:
            page_dims.append((page.rect.width, page.rect.height))
        pdf_document.close()
    except Exception as e:
        st.error(f"Error reading PDF dimensions: {e}")
    pdf_file.seek(0)
    return page_dims

#############################
# Render a specific PDF page to an image
#############################
def render_pdf_page(pdf_bytes, page_idx, dpi=200):
    """
    Renders a specific page of a PDF document into a NumPy image array.
    
    Args:
        pdf_bytes (bytes): The raw byte data of the PDF file.
        page_idx (int): The 0-based index of the page to render.
        dpi (int): The resolution (Dots Per Inch) for rendering. Higher = sharper image.
        
    Returns:
        np.ndarray: An RGB NumPy array representing the rendered page, or None if an error occurs.
    """
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf_document[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_document.close()
        return np.array(img)
    except Exception as e:
        st.error(f"Error rendering PDF page: {e}")
        return None

#############################
# Calculate scaling for bounding boxes
#############################
def calculate_scale_factors(img_width, img_height, pdf_width, pdf_height):
    """
    Calculates the X and Y scaling factors needed to map PDF points to image pixels.
    
    Args:
        img_width (int): Pixel width of the rendered PNG image.
        img_height (int): Pixel height of the rendered PNG image.
        pdf_width (float): Conceptual width of the PDF page in points.
        pdf_height (float): Conceptual height of the PDF page in points.
        
    Returns:
        tuple: (scale_x, scale_y) multipliers.
    """
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height
    return scale_x, scale_y

#############################
# Draw bounding boxes in parallel
#############################
def process_chunks_parallel(chunks_list, img, scale_factors):
    """
    Draws green bounding boxes on the provided image based on normalized coordinate chunks.
    
    Args:
        chunks_list (list): The list of dictionary chunks containing "bboxes" arrays.
        img (np.ndarray): The OpenCV/NumPy image array to draw on.
        scale_factors (tuple): The (scale_x, scale_y) mapping factors from calculate_scale_factors.
        
    Returns:
        np.ndarray: The modified image array with bounding boxes drawn.
    """
    scale_x, scale_y = scale_factors
    img_h, img_w = img.shape[:2]
    
    total_boxes = []
    for chunk in chunks_list:
        for bbox in chunk.get("bboxes", []):
            if len(bbox) == 4:
                # Local format is normalized [x, y, w, h] 0-1
                x, y, w, h = bbox
                
                # Convert to pixel coordinates
                x1 = int(x * img_w)
                y1 = int(y * img_h)
                x2 = int((x + w) * img_w)
                y2 = int((y + h) * img_h)

                # Clip to image boundaries
                x1 = max(0, min(x1, img_w - 1))
                x2 = max(0, min(x2, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                # Only add if box has reasonable size
                if x2 > x1 and y2 > y1 and (x2-x1)*(y2-y1) > 100:  
                    total_boxes.append((x1, y1, x2, y2))

    # Draw boxes with a nice visible GREEN color
    for (x1, y1, x2, y2) in total_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return img

#############################
# Local PDF Parser using PyMuPDF (Replaces agentic-doc)
#############################
def parse_pdf_local(pdf_file, filename):
    """
    Uses PyMuPDF to extract text blocks and their bounding boxes.
    Outputs a structure compatible with the original app design.
    """
    page_map = {}
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            rect = page.rect
            page_w, page_h = rect.width, rect.height
            page_idx = page_num + 1  # 1-based indexing for the app
            
            page_map[page_idx] = []
            
            # Extract blocks of text: (x0, y0, x1, y1, "lines in block", block_no, block_type)
            blocks = page.get_text("blocks")
            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                
                # Only process text blocks (block_type 0)
                if block_type == 0 and text.strip():
                    # Calculate normalized [x, y, w, h]
                    x = x0 / page_w
                    y = y0 / page_h
                    w = (x1 - x0) / page_w
                    h = (y1 - y0) / page_h
                    
                    page_map[page_idx].append({
                        "bboxes": [[x, y, w, h]],
                        "captions": [text.strip()],
                    })
                    
        pdf_document.close()
    except Exception as e:
        st.error(f"Error parsing local PDF '{filename}': {e}")
    finally:
        pdf_file.seek(0)
        
    return page_map

def process_pdfs_with_local(uploaded_pdfs):
    """Process multiple PDFs locally using PyMuPDF"""
    results_map = {} # Maps filename to page_map
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    total = len(uploaded_pdfs)
    for i, pdf_file in enumerate(uploaded_pdfs):
        filename = pdf_file.name
        progress_text.text(f"Extracting layouts from {filename} ({i+1}/{total})...")
        page_map = parse_pdf_local(pdf_file, filename)
        results_map[filename] = page_map
        progress_bar.progress((i + 1) / total)
        
    progress_text.text("Document parsing complete!")
    return results_map

#############################
# Streamlit App
#############################
st.set_page_config(page_title="Multi-PDF Local RAG", page_icon="📄", layout="wide")

st.title("📄 Multi-PDF Local RAG Assistant")
st.markdown("Upload PDFs in the sidebar and chat with your documents using Gemini locally!")

# Main logic for sidebar
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_pdfs:
        st.markdown("### 🔍 PDF Previews")
        for pdf_file in uploaded_pdfs:
            with st.expander(f"Preview: {pdf_file.name}"):
                pdf_bytes = pdf_file.read()
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="300px"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                pdf_file.seek(0)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

all_evidence = {}
all_page_dims = {}

# Process PDFs
if uploaded_pdfs:
    current_pdfs = {pdf.name: pdf for pdf in uploaded_pdfs}

    if ("processed_pdfs" not in st.session_state or current_pdfs.keys() != st.session_state.processed_pdfs.keys()):
        with st.sidebar:
            status_message = st.empty()
            status_message.info(f"Processing {len(uploaded_pdfs)} PDF(s) locally. This is typically very fast...")
            start_time = time.time()
            
            if "raw_pdfs" not in st.session_state:
                st.session_state.raw_pdfs = {}
            
            process_results_map = process_pdfs_with_local(uploaded_pdfs)
            
            for file_info in uploaded_pdfs:
                filename = file_info.name
                page_map = process_results_map.get(filename, {})
                
                # Store extracted text
                for page_num, chunk_list in page_map.items():
                    composite_key = f"{filename}:{page_num}"
                    all_evidence[composite_key] = chunk_list
                    
                # Store minimal required info, avoiding full image creation
                file_info.seek(0)
                st.session_state.raw_pdfs[filename] = file_info.read()
                file_info.seek(0)
                all_page_dims[filename] = get_page_dims(file_info)
                
            elapsed = time.time() - start_time
            status_message.success(f"✅ All PDFs processed locally in {elapsed:.2f} seconds")
            
            st.session_state.all_evidence = all_evidence
            st.session_state.all_page_dims = all_page_dims
            st.session_state.processed_pdfs = current_pdfs
    else:
        all_evidence = st.session_state.all_evidence
        all_page_dims = st.session_state.all_page_dims
else:
    st.info("👈 Please upload PDF files in the sidebar to begin analyzing.")

# Display Chat History
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])
        
        # Display images if the assistant attached evidence
        if "evidence_imgs" in chat and chat["evidence_imgs"]:
            for img, text in chat["evidence_imgs"]:
                st.markdown(text)
                st.image(img, use_container_width=True)
                
        # Display reasoning context inside an expander
        if "reasoning" in chat and chat["reasoning"]:
            with st.expander("💭 Show Reasoning & Details"):
                st.write("**Reasoning Process:**")
                st.write(chat["reasoning"])
                if "best_chunks" in chat and chat["best_chunks"]:
                    st.write("**Extracted Evidence:**")
                    for chunk in chat["best_chunks"]:
                        st.write(f"- _{chunk.get('file')} (Page {chunk.get('page')})_: {chunk.get('reason', '')}")

# 3. Ask a Question
user_query = st.chat_input("Enter your question about the PDFs:")
if user_query and uploaded_pdfs:
    # Append to local history session
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Render user query right away
    with st.chat_message("user"):
        st.write(user_query)

    # Render assistant response as it streams/fetches
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            start_time = time.time()

            filtered_evidence = {k: v for k, v in all_evidence.items() if v}
            combined_evidence = json.dumps(filtered_evidence, indent=2)

            result_json = get_answer_and_best_chunks(user_query, combined_evidence)
            answer = result_json.get("answer", "No answer provided.")
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            best_chunks = result_json.get("best_chunks", [])
            
            st.write(answer)
            evidence_imgs_data = []

            if best_chunks:
                st.markdown("### 🔎 Highlighted Evidence")
                matched = {}
                for chunk in best_chunks:
                    key = f"{chunk.get('file')}:{chunk.get('page')}"
                    matched.setdefault(key, []).append(chunk)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_map = {}
                    for comp_key, chunks_list in matched.items():
                        try:
                            filename, page_str = comp_key.split(":")
                            page_num = int(page_str)
                            page_idx = page_num - 1

                            if filename in st.session_state.raw_pdfs and filename in all_page_dims:
                                pdf_bytes = st.session_state.raw_pdfs[filename]
                                
                                def worker(chunks_list, pdf_bts, p_idx, p_w, p_h):
                                    img = render_pdf_page(pdf_bts, p_idx)
                                    if img is None: return None
                                    scale_f = calculate_scale_factors(img.shape[1], img.shape[0], p_w, p_h)
                                    annotated = process_chunks_parallel(chunks_list, img, scale_f)
                                    return annotated

                                pdf_width, pdf_height = all_page_dims[filename][page_idx]
                                future = executor.submit(
                                    worker, chunks_list, pdf_bytes, page_idx, pdf_width, pdf_height
                                )
                                future_map[future] = (comp_key, filename, page_num)
                        except Exception as e:
                            st.error(f"Error submitting processing task {comp_key}: {e}")

                    for future in concurrent.futures.as_completed(future_map):
                        comp_key, filename, page_num = future_map[future]
                        try:
                            annotated_img = future.result()
                            if annotated_img is not None:
                                text_label = f"**{filename}** (Page {page_num})"
                                st.markdown(text_label)
                                st.image(annotated_img, use_container_width=True)
                                evidence_imgs_data.append((annotated_img, text_label))
                        except Exception as e:
                            st.warning(f"Failed to process {comp_key}: {e}")

            with st.expander("💭 Show Reasoning & Details"):
                st.write("**Reasoning Process:**")
                st.write(reasoning)

                if best_chunks:
                    st.write("**Extracted Evidence:**")
                    for chunk in best_chunks:
                        st.write(f"- _{chunk.get('file')} (Page {chunk.get('page')})_: {chunk.get('reason', '')}")
                else:
                    st.info("No supporting chunks identified by the LLM.")
                
                total_time = round(time.time() - start_time, 2)
                st.write(f"⏱️ _Completed in {total_time} seconds_")

        # Save assistant message to chat history so it persists
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "evidence_imgs": evidence_imgs_data,
            "reasoning": reasoning,
            "best_chunks": best_chunks
        })
