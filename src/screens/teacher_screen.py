import streamlit as st
from src.ui.base_layout import style_background_dashboard,style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from src.components.dialog_create_subject import create_subject_dialog
from src.components.dialog_share_subject import share_subject_dialog
from src.components.subject_card import subject_card
from src.database.db import check_teacher_exists, create_teacher, teacher_login, get_teacher_subjects, get_attendance_for_teacher
from src.components.dialog_add_photo import add_photos_dialog
from src.pipelines.face_pipeline import predict_attendance
import numpy as np
from src.database.config import supabase
from datetime import datetime
import pandas as pd
from src.components.dialog_attendance_results import attendance_result_dialog
from src.components.dialog_voice_attendance import voice_attendance_dialog


def teacher_screen():
    style_background_dashboard()
    style_base_layout()
    if "teacher_data" in st.session_state:
        teacher_dashboard()
    elif 'teacher_login_type' not in st.session_state or st.session_state.teacher_login_type == "login":
        teacher_screen_login()
    elif st.session_state.teacher_login_type == 'register':
        teacher_screen_register()


def teacher_dashboard():
    teacher_data = st.session_state.teacher_data

    c1,c2 = st.columns(2,vertical_alignment='center',gap='xxlarge')
    with c1:
        header_dashboard()
    with c2: 
        st.subheader(f"""Welcome, {teacher_data["name"]}""")
        if st.button("Logout", type='secondary',key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['is_logged_in']= False
            del st.session_state.teacher_data
            st.rerun()
        
    st.space()

    if "current_teacher_tab" not in st.session_state:
        st.session_state.current_teacher_tab = 'take_attendance'

    tab1, tab2, tab3 = st.columns(3)

    with tab1:
        type1 = "primary" if st.session_state.current_teacher_tab == 'take_attendance' else "tertiary"
        if st.button('Take Attendance',type=type1,width='stretch', icon=':material/ar_on_you:'):
            st.session_state.current_teacher_tab = 'take_attendance'
            st.rerun()
    with tab2:
        type2 = "primary" if st.session_state.current_teacher_tab == 'manage_subjects' else "tertiary"
        if st.button('Manage Subjects',type=type2,width='stretch', icon=':material/book_ribbon:'):
            st.session_state.current_teacher_tab = 'manage_subjects'
            st.rerun()
    with tab3:
        type3 = "primary" if st.session_state.current_teacher_tab == 'attendance_records' else "tertiary"
        if st.button('Attendance Records',type=type3,width='stretch', icon=':material/cards_stack:'):
            st.session_state.current_teacher_tab = 'attendance_records'
            st.rerun()

    st.divider()

    if st.session_state.current_teacher_tab == 'take_attendance':
        techer_tab_take_attendance()
    if st.session_state.current_teacher_tab == 'manage_subjects':
        techer_tab_manage_subjects()
    if st.session_state.current_teacher_tab == 'attendance_records':
        teacher_tab_attendance_records()


    footer_dashboard()

def techer_tab_take_attendance():
    teacher_id = st.session_state.teacher_data['teacher_id']
    st.header('📸 Take Attendance')

    # --- 1. SESSION STATE SETUP ---
    if 'attendance_images' not in st.session_state:
        st.session_state.attendance_images = []

    # --- 2. DATA FETCHING ---
    subjects = get_teacher_subjects(teacher_id)

    if not subjects:
        st.warning("🚀 You haven't created any subjects yet. Please create one to begin!")
        return
    
    subject_options = {f"{s['name']} ({s['subject_code']})": s['subject_id'] for s in subjects}

    # --- 3. TOP SELECTION BAR ---
    col1, col2 = st.columns([3, 1], vertical_alignment='bottom')
    with col1:
        selected_subject_label = st.selectbox('Select Subject', options=list(subject_options.keys()))
    with col2:
        # This button triggers the photo upload dialog
        if st.button('Add Photos', type='primary', icon=':material/add_a_photo:', use_container_width=True):
            add_photos_dialog()

    selected_subject_id = subject_options[selected_subject_label]
    st.divider()

    # --- 4. MODE TABS ---
    tab_ai, tab_manual, tab_voice = st.tabs(["🤖 AI Face Recognition", "✍️ Manual Register", "🎤 Voice ID"])

    # --- 5. AI MODE LOGIC ---
    with tab_ai:
        if st.session_state.attendance_images:
            st.subheader('🖼️ Uploaded Session Photos')
            gallery_cols = st.columns(4)
            for idx, img in enumerate(st.session_state.attendance_images):
                with gallery_cols[idx % 4]:
                    st.image(img, use_container_width=True, caption=f'Photo {idx+1}')

            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button('Clear Photos', type='secondary', icon=':material/delete:', use_container_width=True):
                    st.session_state.attendance_images = []
                    st.rerun()
            with c2:
                if st.button('Run AI Analysis', type='primary', icon=':material/analytics:', use_container_width=True):
                    all_detected_ids = {}
                    with st.spinner("AI is analyzing faces..."):
                        for idx, img in enumerate(st.session_state.attendance_images):
                            img_np = np.array(img.convert('RGB'))
                            detected, _, _ = predict_attendance(img_np)
                            if detected:
                                for sid in detected.keys():
                                    student_id = int(sid)
                                    all_detected_ids.setdefault(student_id, []).append(f"Photo {idx+1}")

                    # Process Results
                    enrolled_res = supabase.table('subject_students').select("*,students(*)").eq('subject_id', selected_subject_id).execute()
                    enrolled_students = enrolled_res.data

                    if not enrolled_students:
                        st.error('❌ No students enrolled in this course.')
                    else:
                        results, attendance_to_log = [], []
                        current_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

                        for node in enrolled_students:
                            student = node['students']
                            sources = all_detected_ids.get(int(student['student_id']), [])
                            is_present = len(sources) > 0

                            results.append({
                                "Name": student['name'],
                                "Roll No": student['reg_no'],
                                "Source": ", ".join(sources) if is_present else "-",
                                "Status": "✅ Present" if is_present else "❌ Absent"
                            })
                            attendance_to_log.append({
                                'student_id': student['student_id'],
                                'subject_id': selected_subject_id,
                                'timestamp': current_ts,
                                'is_present': bool(is_present)
                            })
                        attendance_result_dialog(pd.DataFrame(results), attendance_to_log)
        else:
            st.info("💡 Pro Tip: Upload group photos of the class using the 'Add Photos' button above to use AI detection.")

    # --- 6. MANUAL MODE LOGIC (NEW) ---
    with tab_manual:
        st.subheader("✍️ Manual Attendance Sheet")
        
        # Fetch Enrollment specifically for manual mode
        with st.spinner('Fetching student list...'):
            enrolled_res = supabase.table('subject_students').select("*,students(*)").eq('subject_id', selected_subject_id).execute()
            enrolled_students = enrolled_res.data

        if not enrolled_students:
            st.warning('No students found for this subject.')
        else:
            # Prepare and Sort Data
            manual_list = []
            for node in enrolled_students:
                s = node['students']
                manual_list.append({
                    "student_id": s['student_id'],
                    "Reg No": s['reg_no'],
                    "Student Name": s['name'],
                    "Present": True  # Defaulting to true for faster marking
                })
            
            # Sort Reg No in increasing order
            df_manual = pd.DataFrame(manual_list).sort_values(by="Reg No")

            # Attractive Data Editor
            st.write("Toggle the status for each student:")
            edited_df = st.data_editor(
                df_manual,
                column_config={
                    "Present": st.column_config.CheckboxColumn(
                        "Attendance",
                        help="Select to mark present (🔵 Blue Tick)",
                        default=True,
                    ),
                    "student_id": None, # Hide backend ID
                    "Reg No": st.column_config.TextColumn("Registration No", disabled=True),
                    "Student Name": st.column_config.TextColumn("Student Name", disabled=True),
                },
                disabled=["Reg No", "Student Name"],
                hide_index=True,
                use_container_width=True,
                key="manual_attendance_editor"
            )

            # Manual Submit
            if st.button("Save Manual Attendance", type="primary", use_container_width=True, icon=":material/done_all:"):
                current_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                
                # Ensure these names match what is passed to the dialog below
                final_logs = []
                final_summary = []

                for _, row in edited_df.iterrows():
                    final_logs.append({
                        'student_id': row['student_id'],
                        'subject_id': selected_subject_id,
                        'timestamp': current_ts,
                        'is_present': row['Present']
                    })
                    
                    final_summary.append({
                        "Name": row['Student Name'],
                        "Roll No": row['Reg No'],
                        "Source": "Manual Entry",
                        "Status": "✅ Present" if row['Present'] else "❌ Absent"
                    })

                # Both variables now match the lists created above
                attendance_result_dialog(pd.DataFrame(final_summary), final_logs)

    # --- 7. VOICE MODE LOGIC ---
    with tab_voice:
        st.subheader("🎤 Voice Verification")
        st.write("Use the microphone for secure, dual-factor attendance.")
        if st.button('Open Voice Portal', type='primary', use_container_width=True, icon=':material/settings_voice:'):
            voice_attendance_dialog(selected_subject_id)

def techer_tab_manage_subjects():
    teacher_id = st.session_state.teacher_data['teacher_id']
    col1, col2 = st.columns(2)

    with col1:
        st.header('Manage Subjects',width='stretch')
    with col2:
        if st.button('Create New Subjects',width='stretch'):
            create_subject_dialog(teacher_id)

    #List all subjects
    subjects = get_teacher_subjects(teacher_id)
    if not subjects:
        st.info("✨ No subjects found. Let's create your first one above!")
        return
    cols = st.columns(2)
    for index, sub in enumerate(subjects):
            with cols[index % 2]:
                stats = [
                    ("👥","Students",sub['total_students']),
                    ("🕓","Classes",sub['total_classes']),
                ]
                def share_btn():
                    if st.button(f"Share Code: {sub['name']}",key=f"share_{sub['subject_code']}",icon=":material/share:"):
                        share_subject_dialog(sub['name'],sub['subject_code'])

                    st.space()
                subject_card(
                    name = sub['name'],
                    code = sub['subject_code'],
                    section = sub['section'],
                    stats = stats,
                    footer_callback = share_btn
                )


def teacher_tab_attendance_records():
    st.header('Attendance Records')
    teacher_id = st.session_state.teacher_data['teacher_id']
    
    roster_data, logs_data = get_attendance_for_teacher(teacher_id)
    
    if not roster_data:
        st.info("👋 No students assigned to your subjects yet.")
        return

    # Process Roster
    roster_list = [{
        "subject_id": item['subject_id'],
        "Subject": item['subjects']['name'],
        "reg_no": item['students']['reg_no'],
        "student_name": item['students']['name'],
        "student_id": item['students']['student_id']
    } for item in roster_data]
    df_roster = pd.DataFrame(roster_list)

    # UI Selection
    subject_names = sorted(df_roster['Subject'].unique())
    selected_sub = st.selectbox("📚 Select Subject", subject_names)
    
    current_roster = df_roster[df_roster['Subject'] == selected_sub]
    selected_sub_id = current_roster['subject_id'].iloc[0]
    
    df_logs = pd.DataFrame(logs_data)
    if not df_logs.empty:
        df_logs = df_logs[df_logs['subject_id'] == selected_sub_id]

    tab1, tab2 = st.tabs(["🕒 Session Records", "📈 Student Performance"])

    # --- TAB 1: DATE & TIME (BLUE STATUS) ---
    with tab1:
        if df_logs.empty:
            st.warning("No attendance sessions recorded.")
        else:
            df_logs['dt_obj'] = pd.to_datetime(df_logs['timestamp'])
            df_logs['Session Label'] = df_logs['dt_obj'].dt.strftime("%b %d, %Y — %I:%M %p")
            
            session_options = df_logs.sort_values('dt_obj', ascending=False)['Session Label'].unique()
            selected_session = st.selectbox("📅 Select Session", session_options)
            
            session_logs = df_logs[df_logs['Session Label'] == selected_session]
            p_count = session_logs['is_present'].sum()
            t_count = len(current_roster)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Present", f"{p_count}")
            col2.metric("Absent", f"{t_count - p_count}")
            col3.metric("Attendance", f"{(p_count/t_count*100):.1f}%")

            date_view = pd.merge(
                current_roster[['reg_no', 'student_name', 'student_id']],
                session_logs[['student_id', 'is_present']],
                on='student_id', how='left'
            ).fillna(False)
            
            # Use Blue Checkbox emoji for Present, White for Absent
            date_view['Status'] = date_view['is_present'].map({True: "🟦 Present", False: "⬜ Absent"})

            st.dataframe(
                date_view.sort_values(by='reg_no')[['reg_no', 'student_name', 'Status']],
                column_config={"reg_no": "Roll No", "student_name": "Name"},
                use_container_width=True, hide_index=True
            )

    # --- TAB 2: TOTAL PERFORMANCE (WITH TOTAL CLASSES) ---
    with tab2:
        if df_logs.empty:
            st.write("No summary available.")
        else:
            # Calculate total sessions held for this subject
            total_sessions = df_logs['timestamp'].nunique()
            
            # Count presence per student
            stats = df_logs.groupby('student_id')['is_present'].sum().reset_index()
            
            summary = pd.merge(
                current_roster[['reg_no', 'student_name', 'student_id']],
                stats, on='student_id', how='left'
            ).fillna(0)
            
            summary['Total Classes'] = total_sessions
            summary['Percentage'] = (summary['is_present'] / total_sessions * 100).round(1) if total_sessions > 0 else 0
            
            
            st.dataframe(
                summary.sort_values(by='reg_no')[['reg_no', 'student_name', 'is_present', 'Total Classes', 'Percentage']],
                column_config={
                    "reg_no": "Roll No",
                    "student_name": "Student Name",
                    "is_present": "Classes Attended",
                    "Total Classes": "Total Classes",
                    "Percentage": st.column_config.ProgressColumn(
                        "Attendance %",
                        format="%.1f%%",
                        min_value=0, max_value=100,
                    ),
                },
                use_container_width=True, 
                hide_index=True
            )

def login_teacher(username,password):
    if not username or not password:
        return False
    
    teacher = teacher_login(username,password)

    if teacher: 
        st.session_state.user_role ='teacher'
        st.session_state.teacher_data = teacher
        st.session_state.is_logged_in = True
        return True
    
    return False

def teacher_screen_login():
    c1,c2 = st.columns(2,vertical_alignment='center',gap='xxlarge')
    with c1:
        header_dashboard()
    with c2: 
        if st.button("Go Back to Home", type='secondary',key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type']=None
            st.rerun()
    st.header('Login using password',text_alignment='center')
    st.space()
    teacher_username = st.text_input("Enter Username",placeholder="anonyaroy")
    teacher_pass = st.text_input("Enter Password", type='password',placeholder="Enter your password")
    st.divider()
    btnc1, btnc2 = st.columns(2)
    with btnc1:
        if st.button('Login',icon=':material/passkey:',shortcut='control+enter',width='stretch'):
            if login_teacher(teacher_username,teacher_pass):
                st.toast("Welcome back!", icon="👋")
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid username and password combo")
    with btnc2:
        if st.button('Register Instead',type='primary',icon=':material/passkey:',width='stretch'):
            st.session_state.teacher_login_type = 'register'
    footer_dashboard()


def register_teacher(teacher_username,teacher_name,teacher_pass,teacher_pass_confirm):
    if not teacher_username or not teacher_name or not teacher_pass:
        return False, "All Fields are required!"
    if check_teacher_exists(teacher_username):
        return False, "Username already taken"
    if teacher_pass != teacher_pass_confirm:
        return False, "Password doesn't match"
    
    try:
        create_teacher(teacher_username,teacher_pass,teacher_name)
        return True , "Sucessfully Created! Login now"
    except Exception as e:
        return False, "Unexcepted Error!"

def teacher_screen_register():
    c1,c2 = st.columns(2,vertical_alignment='center',gap='xxlarge')
    with c1:
        header_dashboard()
    with c2: 
        if st.button("Go Back to Home", type='secondary',key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type']=None
            st.rerun()
    st.header('Register your teacher profile')
    st.space()
    teacher_username = st.text_input("Enter Username",placeholder="anonyaroy")
    teacher_name = st.text_input("Enter Name",placeholder="anoy roy")
    teacher_pass = st.text_input("Enter Password", type='password',placeholder="Enter password")
    teacher_pass_confirm = st.text_input("Confirm your password",type='password',placeholder="enter password")
    st.divider()
    btnc1, btnc2 = st.columns(2)
    with btnc1:
        if st.button('Register now',icon=':material/passkey:',shortcut='control+enter',width='stretch'):
            success, message = register_teacher(teacher_username,teacher_name,teacher_pass,teacher_pass_confirm)
            if success:
                st.success(message)
                import time
                time.sleep(2)
                st.session_state.teacher_login_type = "login"
                st.rerun()
            else:
                st.error(message)
    with btnc2:
        if st.button('Login Instead',type='primary',icon=':material/passkey:',width='stretch'):
            st.session_state.teacher_login_type = 'login'
    footer_dashboard()
