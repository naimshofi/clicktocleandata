# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import base64
import time
import re

# Page configuration
st.set_page_config(
    page_title="Data Cleaner | Click To Clean",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'clean_steps' not in st.session_state:
    st.session_state.clean_steps = []
if 'backup_dfs' not in st.session_state:
    st.session_state.backup_dfs = []

# Helper functions
def add_clean_step(description):
    """Add a step to cleaning history"""
    st.session_state.clean_steps.append({
        'timestamp': pd.Timestamp.now(),
        'description': description,
        'shape': st.session_state.df.shape
    })

def create_backup():
    """Create backup of current dataframe"""
    if st.session_state.df is not None:
        st.session_state.backup_dfs.append(st.session_state.df.copy())

def check_data_loaded():
    """Check if data is loaded"""
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        return False
    return True

def get_download_link(df, filename="cleaned_data.csv"):
    """Generate download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'

# Main app
def main():
    st.markdown(
        '''
        <style>
        .main-header {
            background-color: #1E3A8A; /* professional deep blue */
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            color: white;
            font-size: 45px;
            font-weight: bold;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )
    # Sidebar navigation
    st.sidebar.title("🧹 CTC Data")
    
    menu_options = [
        "📤 Upload Data",
        "📊 Data Overview",
        "🔍 Missing Values",
        "🔄 Data Types",
        "🗑️ Duplicates",
        "✏️ Column Operations",
        "🔤 String Cleaner",
        "📤 Export Data"
    ]
    
    selected_page = st.sidebar.radio("Navigation", menu_options)
    
    # Show current data info in sidebar
    if st.session_state.df is not None:
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Data Columns:** {st.session_state.df.shape[1]}")
        st.sidebar.info(f"**Data Rows:** {st.session_state.df.shape[0]}")
        st.sidebar.info(f"**Missing Values:** {st.session_state.df.isnull().sum().sum()}")
        
        # Backup/Undo options
        if st.session_state.backup_dfs:
            if st.sidebar.button("↩️ Undo Last Change"):
                st.session_state.df = st.session_state.backup_dfs.pop()
                st.sidebar.success("Last change undone!")
                st.rerun()
    
    # Main content area
    # Main content area
    st.markdown('''<div class="main-header">
                CTC Data | Click To Clean</div>
                </div>''',unsafe_allow_html=True)
    
    # Page routing
    if selected_page == "📤 Upload Data":
        upload_data_page()
    elif selected_page == "📊 Data Overview":
        if check_data_loaded():
            data_overview_page()
    elif selected_page == "🔍 Missing Values":
        if check_data_loaded():
            missing_values_page()
    elif selected_page == "🔄 Data Types":
        if check_data_loaded():
            data_types_page()
    elif selected_page == "🗑️ Duplicates":
        if check_data_loaded():
            duplicates_page()
    elif selected_page == "✏️ Column Operations":
        if check_data_loaded():
            column_operations_page()
    elif selected_page == "🔤 String Cleaner":
        if check_data_loaded():
            string_cleaner_page()
    elif selected_page == "📤 Export Data":
        if check_data_loaded():
            export_data_page()

# ====================================
# PAGE 1: Upload Data
# ====================================
def upload_data_page():
    st.header("📤 Upload Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Upload your file')

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'txt'],
            help="Supported formats: CSV, Excel, JSON, TXT"
        )
        
        if uploaded_file is not None:
            try:
                # Read file based on extension
                file_ext = uploaded_file.name.split('.')[-1].lower()
                
                if file_ext == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_ext in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                elif file_ext == 'json':
                    df = pd.read_json(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, sep=None, engine='python')
                
                # Store in session state
                st.session_state.df = df
                st.session_state.original_df = df.copy()
                st.session_state.clean_steps = []
                st.session_state.backup_dfs = []
                
                st.success(f"✅ Data uploaded and loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
            
        st.subheader('Or Use a Link')

        data_url = st.text_input(
            'Enter dataset URL',
            placeholder='https://example.com/data.csv'
        )
        
        upl_btn = st.button('Upload')            
        
        if upl_btn and data_url:
            try:
                if data_url.endswith('.csv'):
                    df = pd.read_csv(data_url)
                elif data_url.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(data_url)
                elif data_url.endswith('.json'):
                    df = pd.read_json(data_url)
                else:
                    df = pd.read_csv(data_url, sep=None, engine='python')

                # Store in session state
                st.session_state.df = df
                st.session_state.original_df = df.copy()
                st.session_state.clean_steps = []
                st.session_state.backup_dfs = []
                st.success('✅ Data loaded successfully from link!')

            except Exception as e:
                st.error(f'Error loading link: {str(e)}')
    
    with col2:
        st.subheader("📋 Quick Start")
        st.markdown("""
        1. Upload your dataset
        2. Explore data overview
        3. Use cleaning modules
        4. Export cleaned data
        """)
        
        # Sample data option
        if st.button("Try with Sample Data"):
            # Create sample data
            sample_df = pd.DataFrame({
                'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown','Naim','Naim'],
                'Age': [25, 30, None, 35, 40, 35, 35],
                'Salary': [50000, None, 55000, 70000, 80000, 50000, 50000],
                'Department': ['IT', 'HR', 'IT', 'Finance', None, 'IT', 'IT'],
                'Join_Date': ['2020-01-15', '2019-03-20', '2021-06-10', None, '2018-11-05','2025-08-30','2025-08-30'],
                'Email': ['john@email.com', 'jane@email.com', 'bob@email  .com', 'finance@email.COM', None, 'naim_38@email.com','naim_38@email.com']
            })
            
            st.session_state.df = sample_df
            st.session_state.original_df = sample_df.copy()
            st.session_state.clean_steps = []
            st.session_state.backup_dfs = []
            st.success("✅ Sample data loaded!")
            #st.rerun()
        
        # Reset to original
        if st.session_state.original_df is not None:
            if st.button("🔄 Reset to Original Data"):
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state.clean_steps = []
                st.session_state.backup_dfs = []
                st.success("Data reset to original!")
                #st.rerun()

# ====================================
# PAGE 2: Data Overview
# ====================================
def data_overview_page():
    st.header("📊 Data Overview")
    
    df = st.session_state.df
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        '🔍 Preview', 
        '📈 Summary', 
        '📋 Data Types', 
        '❌ Missing Values'        
    ])
    
    with tab1:
        # Data preview with pagination
        rows_per_page = st.slider('Rows per page', 5, 100, 10, key='preview_slider')
        page_number = st.number_input('Page', 1, max(1, len(df)//rows_per_page + 1), 1)
        
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        st.dataframe(df.iloc[start_idx:end_idx], width='stretch')
        
        # Show all data option
        if st.checkbox('Show all data'):
            st.dataframe(df, width='stretch')
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include='all'), width='stretch')
    
    with tab3:
        st.subheader("Data Types & Info")
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': df.nunique().values
        })
        st.dataframe(dtype_info, width='stretch')
    
    with tab4:
        st.subheader("Missing Values Analysis")
        
        missing_info = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        }).sort_values('Missing %', ascending=False)
        
        st.dataframe(missing_info, width='stretch')
        
        # Visualization
        if missing_info['Missing Count'].sum() > 0:
            fig = px.bar(missing_info[missing_info['Missing Count'] > 0], 
                        x='Column', y='Missing Count',
                        title='Missing Values by Column',
                        color='Missing %',
                        color_continuous_scale='Reds')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width='stretch')

# ====================================
# PAGE 3: Missing Values
# ====================================
def missing_values_page():
    st.header("🔍 Missing Values Handler")
    
    df = st.session_state.df
    
    # Display current missing values
    st.subheader("📈 Current Missing Values Summary")

    # Create a more informative dataframe
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
        'Unique Values': df.nunique().values
    })
    
    # Sort by missing percentage
    missing_df = missing_df.sort_values('Missing %', ascending=False)
    
    # Display with formatting
    styled_df = missing_df.style.background_gradient(
        subset=['Missing %'], 
        cmap='Reds', 
        vmin=0, 
        vmax=100
    ).format({
        'Missing %': '{:.2f}%',
        'Missing Count': '{:.0f}'
    })
    
    missing_df_row = len(missing_df)*40       
    st.dataframe(styled_df, width='stretch', height=missing_df_row)

    missing_total = df.isnull().sum().sum()
    st.info(f"Total missing values: **{missing_total}**")

    # Get missing column and separate it to use for selection
    missing_cols = df.columns[df.isnull().any()].tolist()
    num_cols_missing = [col for col in missing_cols if pd.api.types.is_numeric_dtype(df[col])]
    non_num_cols_missing = [col for col in missing_cols if not pd.api.types.is_numeric_dtype(df[col])]

    if missing_total == 0:
        st.success("🎉 No missing values found in the dataset!")
        return
    
    # Method selection
    st.subheader("👨‍🔬 Treatment Methods")
    method = st.selectbox(
        "Select treatment method:",
        ["Fill with Value", "Fill with Statistics", "Forward/Backward Fill", "Drop Rows/Columns"]
    )
    
    if method == "Fill with Value":
        col1, col2 = st.columns(2)
        
        with col1:
            columns_to_fill = st.multiselect(
                "Select columns to fill:",
                missing_cols,
                default=[col for col in df.columns if df[col].isnull().any()][:3]
            )
        
        with col2:
            fill_value = st.text_input("Fill value:", "Unknown")
        
        if st.button("Apply Fill", type="primary"):
            if columns_to_fill:
                create_backup()
                for col in columns_to_fill:
                    # Try to convert fill value to appropriate type
                    try:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fill_val = float(fill_value)
                        else:
                            fill_val = fill_value
                        st.session_state.df[col] = st.session_state.df[col].fillna(fill_val)
                    except:
                        st.session_state.df[col] = st.session_state.df[col].fillna(fill_value)
                
                add_clean_step(f"Filled missing values in {len(columns_to_fill)} columns with '{fill_value}'")
                st.success("✅ Missing values filled!")
                time.sleep(2)
                st.rerun()
    
    elif method == "Fill with Statistics":
        col1, col2 = st.columns(2)
        
        with col1:
            columns_to_fill = st.multiselect(
                "Select numerical columns:",
                num_cols_missing,
                default=num_cols_missing[:min(3, len(num_cols_missing))]
            )
        
        with col2:
            stat_method = st.selectbox(
                "Statistical method:",
                ["Mean", "Median", "Mode", "Zero"]
            )
        
        if st.button("Apply Statistical Fill", type="primary"):
            if columns_to_fill:
                create_backup()
                for col in columns_to_fill:
                    if stat_method == "Mean":
                        fill_val = df[col].mean()
                    elif stat_method == "Median":
                        fill_val = df[col].median()
                    elif stat_method == "Mode":
                        fill_val = df[col].mode()[0] if not df[col].mode().empty else 0
                    else:  # Zero
                        fill_val = 0
                    
                    st.session_state.df[col] = st.session_state.df[col].fillna(fill_val)
                
                add_clean_step(f"Filled {len(columns_to_fill)} columns with {stat_method}")
                st.success("✅ Missing values filled with statistics!")
                time.sleep(2)
                st.rerun()
    
    elif method == "Forward/Backward Fill":
        col1, col2 = st.columns(2)
        
        with col1:
            columns_to_fill = st.multiselect(
                "Select columns:",
                missing_cols,
                default=[col for col in df.columns if df[col].isnull().any()][:3]
            )
        
        with col2:
            fill_direction = st.selectbox(
                "Fill direction:",
                ["Forward Fill (ffill)", "Backward Fill (bfill)"]
            )
        
        if st.button("Apply Forward/Backward Fill", type="primary"):
            if columns_to_fill:
                create_backup()
                for col in columns_to_fill:
                    if fill_direction == "Forward Fill (ffill)":
                        st.session_state.df[col] = st.session_state.df[col].fillna(method='ffill')
                    else:
                        st.session_state.df[col] = st.session_state.df[col].fillna(method='bfill')
                
                add_clean_step(f"Applied {fill_direction} to {len(columns_to_fill)} columns")
                st.success("✅ Forward/backward fill applied!")
                time.sleep(2)
                st.rerun()
    
    elif method == "Drop Rows/Columns":
        drop_option = st.radio(
            "Drop option:",
            ["Drop rows with any missing", "Drop rows with all missing", "Drop columns with missing"]
        )
        
        if drop_option == "Drop columns with missing":
            threshold = st.slider(
                "Drop columns with missing % above:",
                0, 100, 50, 5,
                format="%d%%"
            )
            
            cols_to_drop = []
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct >= threshold:
                    cols_to_drop.append(col)
            
            if cols_to_drop:
                st.warning(f"Columns to drop: {', '.join(cols_to_drop)}")
        
        if st.button("Apply Drop", type="primary", help="This action cannot be undone"):
            create_backup()
            
            if drop_option == "Drop rows with any missing":
                rows_before = len(st.session_state.df)
                st.session_state.df = st.session_state.df.dropna()
                rows_after = len(st.session_state.df)
                add_clean_step(f"Dropped rows with any missing values ({rows_before} → {rows_after} rows)")
                
            elif drop_option == "Drop rows with all missing":
                rows_before = len(st.session_state.df)
                st.session_state.df = st.session_state.df.dropna(how='all')
                rows_after = len(st.session_state.df)
                add_clean_step(f"Dropped rows with all missing values ({rows_before} → {rows_after} rows)")
                
            else:  # Drop columns
                cols_before = len(st.session_state.df.columns)
                cols_to_keep = [col for col in st.session_state.df.columns if col not in cols_to_drop]
                st.session_state.df = st.session_state.df[cols_to_keep]
                cols_after = len(st.session_state.df.columns)
                add_clean_step(f"Dropped {cols_before - cols_after} columns with high missing %")
            
            st.success("✅ Drop operation completed!")
            time.sleep(2)
            st.rerun()

# ====================================
# PAGE 4: Data Types
# ====================================
def data_types_page():
    st.header("🔄 Data Type Converter")
    
    df = st.session_state.df
    
    # Display current data types
    st.subheader("Current Data Types")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Current Type': df.dtypes.astype(str),
        'Sample Values': df.iloc[0].astype(str) if len(df) > 0 else [''] * len(df.columns)
    })
    st.dataframe(dtype_df, width="stretch")
    
    # Type conversion
    st.subheader("Convert Data Types")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        columns_to_convert = st.multiselect(
            "Select columns:",
            df.columns.tolist()
        )
    
    with col2:
        target_type = st.selectbox(
            "Convert to:",
            ["String", "Integer", "Float", "Boolean", "Datetime", "Category"]
        )
    
    with col3:
        st.write("Options")
        if target_type == "Datetime":
            date_format = st.text_input("Date format (optional):", help="e.g., %Y-%m-%d")
        elif target_type == "Integer" or target_type == "Float":
            handle_errors = st.checkbox("Set errors to NaN", value=True)
    
    if st.button("Convert Data Types", type="primary"):
        if columns_to_convert:
            create_backup()
            
            for col in columns_to_convert:
                try:
                    if target_type == "String":
                        st.session_state.df[col] = st.session_state.df[col].astype(str)
                    
                    elif target_type == "Integer":
                        if handle_errors:
                            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce').astype('Int64')
                        else:
                            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce').fillna(0).astype(int)
                    
                    elif target_type == "Float":
                        if handle_errors:
                            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce')
                        else:
                            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce').fillna(0.0)
                    
                    elif target_type == "Boolean":
                        st.session_state.df[col] = st.session_state.df[col].astype(bool)
                    
                    elif target_type == "Datetime":
                        if date_format:
                            st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], format=date_format, errors='coerce')
                        else:
                            st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce')
                    
                    elif target_type == "Category":
                        st.session_state.df[col] = st.session_state.df[col].astype('category')
                
                except Exception as e:
                    st.warning(f"Could not convert column '{col}': {str(e)}")
            
            add_clean_step(f"Converted {len(columns_to_convert)} columns to {target_type}")
            st.success("✅ Data types converted!")
            time.sleep(2)
            st.rerun()

# ====================================
# PAGE 5: Duplicates
# ====================================
def duplicates_page():
    st.header("🗑️ Duplicate Manager")
    
    df = st.session_state.df
    
    # Find duplicates
    duplicate_count = df.duplicated().sum()
    st.subheader(f"Found {duplicate_count} duplicate rows")
    
    if duplicate_count > 0:
        st.dataframe(df[df.duplicated()].head(), width="stretch")
    
    # Duplicate handling options
    st.subheader("Duplicate Handling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duplicate_method = st.radio(
            "Action:",
            ["Remove all duplicates", "Keep first occurrence", "Keep last occurrence"]
        )
    
    with col2:
        if st.button("Process Duplicates", type="primary"):
            create_backup()
            
            if duplicate_method == "Remove all duplicates":
                rows_before = len(st.session_state.df)
                st.session_state.df = st.session_state.df.drop_duplicates(keep=False)
                rows_after = len(st.session_state.df)
                removed = rows_before - rows_after
                add_clean_step(f"Removed all duplicate rows ({removed} rows removed)")
            
            elif duplicate_method == "Keep first occurrence":
                rows_before = len(st.session_state.df)
                st.session_state.df = st.session_state.df.drop_duplicates( keep='first')
                rows_after = len(st.session_state.df)
                removed = rows_before - rows_after
                add_clean_step(f"Kept first occurrence of duplicates ({removed} rows removed)")
            
            elif duplicate_method == "Keep last occurrence":
                rows_before = len(st.session_state.df)
                st.session_state.df = st.session_state.df.drop_duplicates( keep='last')
                rows_after = len(st.session_state.df)
                removed = rows_before - rows_after
                add_clean_step(f"Kept last occurrence of duplicates ({removed} rows removed)")
            
            st.success("✅ Duplicate processing completed!")
            time.sleep(2)
            st.rerun()
    
    

# ====================================
# PAGE 6: Column Operations
# ====================================
def column_operations_page():
    st.header("✏️ Column Operations")
    
    df = st.session_state.df
    
    # Column operations
    st.subheader("Column Management")
    
    operation = st.selectbox(
        "Select operation:",
        ["Rename Columns", "Drop Columns", "Reorder Columns", "Create New Column"]
    )
    
    if operation == "Rename Columns":
        col1, col2 = st.columns(2)
        
        with col1:
            column_to_rename = st.selectbox("Select column:", df.columns.tolist())
        
        with col2:
            new_name = st.text_input("New column name:", value=column_to_rename)
        
        if st.button("Rename Column", type="primary"):
            if new_name and new_name != column_to_rename:
                create_backup()
                st.session_state.df = st.session_state.df.rename(columns={column_to_rename: new_name})
                add_clean_step(f"Renamed column '{column_to_rename}' to '{new_name}'")
                st.success("✅ Column renamed!")
                time.sleep(2)
                st.rerun()
    
    elif operation == "Drop Columns":
        columns_to_drop = st.multiselect(
            "Select columns to drop:",
            df.columns.tolist()
        )
        
        if st.button("Drop Columns", type="primary"):
            if columns_to_drop:
                create_backup()
                cols_before = len(st.session_state.df.columns)
                st.session_state.df = st.session_state.df.drop(columns=columns_to_drop)
                cols_after = len(st.session_state.df.columns)
                add_clean_step(f"Dropped {cols_before - cols_after} columns: {', '.join(columns_to_drop)}")
                st.success("✅ Columns dropped!")
                time.sleep(2)
                st.rerun()
    
    elif operation == "Reorder Columns":
        current_order = df.columns.tolist()
        st.write("Current order:", ", ".join(current_order))
        
        new_order = st.multiselect(
            "Select columns in new order:",
            current_order,
            default=current_order
        )
        
        if st.button("Reorder Columns", type="primary"):
            if new_order:
                create_backup()
                # Add any missing columns
                missing_cols = [col for col in current_order if col not in new_order]
                new_order = new_order + missing_cols
                st.session_state.df = st.session_state.df[new_order]
                add_clean_step("Reordered columns")
                st.success("✅ Columns reordered!")
                time.sleep(2)
                st.rerun()
    
    elif operation == "Create New Column":
        col1, col2 = st.columns(2)
        
        with col1:
            new_col_name = st.text_input("New column name:")
        
        with col2:
            col_type = st.selectbox("Column type:", ["Constant", "From Calculation", "From Other Columns"])
        
        if col_type == "Constant":
            constant_value = st.text_input("Constant value:", "0")
            
            if st.button("Create Column", type="primary"):
                if new_col_name:
                    create_backup()
                    st.session_state.df[new_col_name] = constant_value
                    add_clean_step(f"Created new column '{new_col_name}' with constant value")
                    st.success("✅ Column created!")
                    time.sleep(2)
                    st.rerun()
        
        elif col_type == "From Calculation":
            calc_type = st.selectbox("Calculation:", ["Row number", "Length of text"])
            
            if st.button("Create Column", type="primary"):
                if new_col_name:
                    create_backup()
                    if calc_type == "Row number":
                        st.session_state.df[new_col_name] = range(1, len(st.session_state.df) + 1)
                    elif calc_type == "Length of text":
                        # Apply to first text column found
                        text_cols = st.session_state.df.select_dtypes(include=['object']).columns
                        if len(text_cols) > 0:
                            st.session_state.df[new_col_name] = st.session_state.df[text_cols[0]].astype(str).str.len()
                        else:
                            st.warning("No text columns found for length calculation")
                    add_clean_step(f"Created new column '{new_col_name}' from calculation")
                    st.success("✅ Column created!")
                    time.sleep(2)
                    st.rerun()
        
        elif col_type == "From Other Columns":
            col1_sel, col2_sel = st.columns(2)
            
            with col1_sel:
                col_a = st.selectbox("First column:", df.columns.tolist())
            
            with col2_sel:
                operation = st.selectbox("Operation:", ["Concatenate", "Add", "Subtract", "Multiply", "Divide"])
                if operation == "Concatenate":
                    col_b = st.selectbox("Second column:", df.columns.tolist())
                else:
                    # For numeric operations, filter numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    col_b = st.selectbox("Second column:", numeric_cols)
            
            if st.button("Create Column", type="primary"):
                if new_col_name and col_a and col_b:
                    create_backup()
                    try:
                        if operation == "Concatenate":
                            st.session_state.df[new_col_name] = st.session_state.df[col_a].astype(str) + " " + st.session_state.df[col_b].astype(str)
                        elif operation == "Add":
                            st.session_state.df[new_col_name] = st.session_state.df[col_a] + st.session_state.df[col_b]
                        elif operation == "Subtract":
                            st.session_state.df[new_col_name] = st.session_state.df[col_a] - st.session_state.df[col_b]
                        elif operation == "Multiply":
                            st.session_state.df[new_col_name] = st.session_state.df[col_a] * st.session_state.df[col_b]
                        elif operation == "Divide":
                            st.session_state.df[new_col_name] = st.session_state.df[col_a] / st.session_state.df[col_b]
                        add_clean_step(f"Created new column '{new_col_name}' from {col_a} and {col_b}")
                        st.success("✅ Column created!")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating column: {str(e)}")

# ====================================
# PAGE 7: String Cleaner
# ====================================
def string_cleaner_page():
    st.header("🔤 String Cleaner")
    
    df = st.session_state.df
    
    # Select string columns
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not string_cols:
        st.info("No string columns found in the dataset.")
        return
    
    st.subheader("Select Columns to Clean")
    columns_to_clean = st.multiselect(
        "String columns:",
        string_cols,
        default=string_cols[:min(3, len(string_cols))]
    )
    
    if not columns_to_clean:
        return
    
    # String operations
    st.subheader("String Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        remove_whitespace = st.checkbox("Remove extra whitespace", value=True)
        lowercase = st.checkbox("Convert to lowercase")
        uppercase = st.checkbox("Convert to uppercase")
        custom_regex = st.checkbox("Use custom regex")        
    
    with col2:
        remove_special = st.checkbox("Remove special characters")
        remove_digits = st.checkbox("Remove digits")
        remove_punctuation = st.checkbox("Remove punctuation")
        if custom_regex:
            input_regex = st.text_input('Type regex :',value=r"", placeholder="(\\d{4})")
    
    with col3:
        trim_spaces = st.checkbox("Trim spaces")
        replace_empty = st.checkbox("Replace empty strings with NaN")
        fix_email = st.checkbox("Fix email format")
    
    if st.button("Clean Strings", type="primary"):
        create_backup()
        
        for col in columns_to_clean:
            # Create a copy of the column
            cleaned = st.session_state.df[col].astype(str).copy()
            
            # Apply transformations
            if remove_whitespace:
                cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
            
            if trim_spaces:
                cleaned = cleaned.str.strip()
            
            if lowercase:
                cleaned = cleaned.str.lower()
            
            if uppercase:
                cleaned = cleaned.str.upper()

            try:
                if custom_regex and input_regex:
                    cleaned = cleaned.str.extract(input_regex, expand=False)
            except Exception as e:
                st.error(f"Error converting: {str(e)}")
            
            if remove_special:
                # Keep only alphanumeric and spaces
                cleaned = cleaned.str.replace(r'[^A-Za-z0-9\s]', '', regex=True)
            
            if remove_digits:
                cleaned = cleaned.str.replace(r'\d+', '', regex=True)
            
            if remove_punctuation:
                import string
                cleaned = cleaned.str.translate(str.maketrans('', '', string.punctuation))
            
            if fix_email:
                # Basic email cleanup
                cleaned = cleaned.str.replace(r'\s+', '', regex=True)  # Remove spaces in emails
                cleaned = cleaned.str.lower()  # Standardize case
            
            if replace_empty:
                cleaned = cleaned.replace(['', 'nan', 'None', 'null'], np.nan)
            
            # Update the dataframe
            st.session_state.df[col] = cleaned
        
        add_clean_step(f"Cleaned strings in {len(columns_to_clean)} columns")
        st.success("✅ String cleaning completed!")
        time.sleep(2)
        st.rerun()

# ====================================
# PAGE 8: Export Data
# ====================================
def export_data_page():
    st.header("📤 Export Cleaned Data")
    
    df = st.session_state.df
    
    # Data summary
    st.subheader("Final Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Show final data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(), width="stretch")
    
    # Cleaning history
    if st.session_state.clean_steps:
        st.subheader("Cleaning History")
        history_df = pd.DataFrame(st.session_state.clean_steps)
        st.dataframe(history_df, width="stretch")
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file_format = st.selectbox(
            "Export format:",
            ["CSV", "Excel", "JSON"]
        )
        
        include_index = st.checkbox("Include index", value=False)
    
    with col2:
        filename = st.text_input("Filename:", "cleaned_data")
        
        # Encoding option
        encoding = st.selectbox(
            "Encoding:",
            ["utf-8", "latin-1", "utf-16"]
        )
    
    # Export button
    if st.button("📥 Export Data", type="primary"):
        buffer = BytesIO()
        
        try:
            if file_format == "CSV":
                csv = df.to_csv(index=include_index, encoding=encoding)
                buffer.write(csv.encode(encoding))
                mime_type = "text/csv"
                file_ext = "csv"
            
            elif file_format == "Excel":
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=include_index, sheet_name='Cleaned Data')
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                file_ext = "xlsx"
            
            else:  # JSON
                json_str = df.to_json(orient='records', indent=2)
                buffer.write(json_str.encode(encoding))
                mime_type = "application/json"
                file_ext = "json"
            
            buffer.seek(0)
            
            # Create download link
            b64 = base64.b64encode(buffer.read()).decode()
            download_link = f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{file_ext}">Click here to download {filename}.{file_ext}</a>'
            
            st.markdown(download_link, unsafe_allow_html=True)
            st.success("✅ Export ready!")
            
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()