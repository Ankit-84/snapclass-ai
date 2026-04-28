import streamlit as st
from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from PIL import Image
from src.pipelines.face_pipeline import predict_attendance, get_face_embeddings, train_classifier
from src.pipelines.voice_pipeline import get_voice_embedding
import numpy as np
from src.database.db import get_all_students, create_student,get_student_subjects, get_student_attendance, unenroll_student_to_subject
import time
from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card

def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data['student_id']

    # 1. SIMPLE & ATTRACTIVE HEADER
    with st.container():
        c1, c2 = st.columns([3, 1], vertical_alignment='center')
        with c1:
            st.title(f"👋 Welcome, {student_data['name']}")
            st.caption(f"🆔 Registration No: **{student_data['reg_no']}**")
        with c2:
            if st.button("Logout", type='secondary', use_container_width=True, icon=":material/logout:"):
                st.session_state['is_logged_in'] = False
                del st.session_state.student_data
                st.rerun()

    st.divider()

    # 2. DATA FETCHING & PROCESSING
    with st.spinner('Updating your attendance...'):
        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    stats_map = {}
    for log in logs:
        sid = log['subject_id']
        stats_map.setdefault(sid, {"total": 0, "attended": 0})
        stats_map[sid]['total'] += 1
        if log.get('is_present'):
            stats_map[sid]['attended'] += 1

    # 4. RESPONSIVE SUBJECTS SECTION
    sc1, sc2 = st.columns([3, 1], vertical_alignment='bottom')
    with sc1:
        st.header('📚 Your Courses')
    with sc2:
        if st.button('Enroll New', type='primary', use_container_width=True, icon=":material/add_circle:"):
            enroll_dialog()

    if not subjects:
        st.info("You haven't enrolled in any subjects yet. Click 'Enroll New' to begin!")
    else:
        # Create a 2-column responsive grid for subjects
        cols = st.columns(2)
        for i, sub_node in enumerate(subjects):
            sub = sub_node['subjects']
            sid = sub['subject_id']
            stats = stats_map.get(sid, {"total": 0, "attended": 0})

            # FIX: Closure with unique keys
            def make_unenroll_callback(s_id=sid, s_name=sub['name']):
                if st.button("Unenroll", key=f"un_{s_id}", type='tertiary', icon=':material/close:', use_container_width=True):
                    unenroll_student_to_subject(student_id, s_id)
                    st.toast(f'Unenrolled from {s_name}!')
                    time.sleep(0.5)
                    st.rerun()

            with cols[i % 2]:
                subject_card(
                    name=sub['name'],
                    code=sub['subject_code'],
                    section=sub['section'],
                    stats=[
                        ("🗓️", "Total", stats['total']),
                        ("✅", "Attended", stats['attended']),
                    ],
                    footer_callback=make_unenroll_callback
                )

    footer_dashboard()

def student_screen():
    style_background_dashboard()
    style_base_layout()

    # 1. Check if already logged in
    if "student_data" in st.session_state:
        student_dashboard()
        return

    # 2. Navigation Header
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go Back to Home", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()

    st.header('Login using FaceID', anchor=False)

    if 'show_registration' not in st.session_state:
        st.session_state.show_registration = False

    photo_source = st.camera_input("Position your face in the center")

    if photo_source:
        img = np.array(Image.open(photo_source))

        if not st.session_state.show_registration:
            with st.spinner('AI is scanning...'):
                detected, all_ids, num_faces = predict_attendance(img)

                if num_faces == 0:
                    st.warning('Face not found!')
                elif num_faces > 1:
                    st.warning('Multiple faces found. Please ensure only one person is in frame.')
                else:
                    if detected:
                        student_id = list(detected.keys())[0]
                        all_students = get_all_students()
                        student = next((s for s in all_students if s['student_id'] == student_id), None)

                        if student:
                            st.session_state.is_logged_in = True
                            st.session_state.user_role = 'student'
                            st.session_state.student_data = student
                            st.toast(f"Welcome Back {student['name']}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.info('Face not recognized! You might be a new student.')
                            st.session_state.show_registration = True
                    else:
                        st.info('Face not recognized! Please register below.')
                        st.session_state.show_registration = True

    # 3. Registration Section
    if st.session_state.show_registration and photo_source:
        with st.container(border=True):
            st.header('Register New Profile')
            new_name = st.text_input("Enter your name", placeholder='E.g. Ankit Kumar')
            reg_no = st.text_input("Enter your registration no",placeholder='E.g. 2024105357')

            st.subheader('Optional: Voice Enrollment')
            st.info("Enroll your voice for dual-factor attendance.")

            audio_data = None
            try:
                audio_data = st.audio_input('Record: "I am present, my name is..."')
            except Exception as e:
                st.error(f'Audio hardware error: {e}')

            if st.button('Create Account', type='primary'):
                if new_name:
                    with st.spinner('Creating profile...'):
                        # Re-process image for embedding
                        img_to_process = np.array(Image.open(photo_source))
                        encodings = get_face_embeddings(img_to_process)
                        
                        if encodings:
                            face_emb = encodings[0].tolist()
                            voice_emb = None
                            
                            if audio_data:
                                voice_emb = get_voice_embedding(audio_data.read())

                            response_data = create_student(new_name,reg_no, face_embedding=face_emb, voice_embedding=voice_emb)

                            if response_data:
                                train_classifier()
                                st.session_state.is_logged_in = True
                                st.session_state.user_role = 'student'
                                st.session_state.student_data = response_data[0]
                                st.session_state.show_registration = False # Reset state
                                st.toast(f'Profile Created! Welcome {new_name}!')
                                time.sleep(1)
                                st.rerun()
                        else:
                            st.error("Couldn't capture facial features. Try adjusting lighting.")
                else:
                    st.warning('Please enter your name!')

    footer_dashboard()