import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Shopify Analytics Dashboard", layout="wide")

st.title("📊 Shopify Analytics Dashboard")

def parse_dates(df, date_column):
    """Parse dates with multiple format attempts for robust handling"""
    try:
        df[date_column] = pd.to_datetime(df[date_column], format='mixed', errors='coerce')
    except:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    if df[date_column].isna().any():
        st.warning(f"Warning: {df[date_column].isna().sum()} dates could not be parsed and will be excluded from analysis.")
    
    return df

def calculate_cohort_retention(df):
    """Calculate monthly cohort retention percentages and cohort sizes"""
    df_clean = df.dropna(subset=['Created at', 'Customer ID'])
    
    df_clean['Order Month'] = df_clean['Created at'].dt.to_period('M')
    
    cohort_data = df_clean.groupby('Customer ID')['Order Month'].agg(['min', 'count']).reset_index()
    cohort_data.columns = ['Customer ID', 'Cohort Month', 'Total Orders']
    
    df_with_cohort = df_clean.merge(cohort_data[['Customer ID', 'Cohort Month']], on='Customer ID')
    
    df_with_cohort['Months Since First Purchase'] = (
        (df_with_cohort['Order Month'] - df_with_cohort['Cohort Month']).apply(lambda x: x.n)
    )
    
    cohort_counts = df_with_cohort.groupby(['Cohort Month', 'Months Since First Purchase'])['Customer ID'].nunique().reset_index()
    cohort_counts.columns = ['Cohort Month', 'Months Since First Purchase', 'Customer Count']
    
    cohort_pivot = cohort_counts.pivot_table(
        index='Cohort Month',
        columns='Months Since First Purchase',
        values='Customer Count',
        fill_value=0
    )
    
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100
    
    return retention_matrix, cohort_size

def calculate_customer_ltv(df):
    """Calculate customer lifetime value metrics"""
    df_clean = df.dropna(subset=['Customer ID', 'Total', 'Created at'])
    
    customer_metrics = df_clean.groupby('Customer ID').agg({
        'Total': ['sum', 'mean', 'count'],
        'Created at': ['min', 'max']
    }).reset_index()
    
    customer_metrics.columns = ['Customer ID', 'Total Spend', 'Avg Order Value', 'Order Count', 'First Purchase', 'Last Purchase']
    
    customer_metrics['Lifespan Months'] = ((customer_metrics['Last Purchase'] - customer_metrics['First Purchase']).dt.days / 30.44)
    customer_metrics['Lifespan Months'] = customer_metrics['Lifespan Months'].apply(lambda x: max(x, 1))
    
    customer_metrics['Purchase Frequency'] = customer_metrics['Order Count'] / customer_metrics['Lifespan Months']
    customer_metrics['LTV'] = customer_metrics['Avg Order Value'] * customer_metrics['Purchase Frequency'] * 12
    
    return customer_metrics

def get_whale_customers(df):
    """Get top 10 customers by total spend"""
    df_clean = df.dropna(subset=['Customer ID', 'Total', 'Email'])
    
    whale_customers = df_clean.groupby(['Customer ID', 'Email']).agg({
        'Total': 'sum',
        'Order ID': 'count'
    }).reset_index()
    
    whale_customers.columns = ['Customer ID', 'Email', 'Total Spend', 'Order Count']
    whale_customers = whale_customers.sort_values('Total Spend', ascending=False).head(10)
    
    return whale_customers

def create_pdf_report(retention_data, whale_data, cohort_size_data):
    """Generate PDF report with retention and whale customer data"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Shopify Analytics Report", styles['Title']))
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Paragraph("Cohort Retention Matrix (%)", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    retention_data_str = retention_data.copy()
    retention_data_str.index = retention_data_str.index.astype(str)
    retention_table_data = [['Cohort'] + [str(col) for col in retention_data_str.columns]]
    for idx, row in retention_data_str.iterrows():
        retention_table_data.append([idx] + [f"{val:.1f}" for val in row])
    
    retention_table = Table(retention_table_data)
    retention_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(retention_table)
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Paragraph("Whale Customers (Top 10)", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    whale_table_data = [['Customer ID', 'Email', 'Total Spend', 'Order Count']]
    for _, row in whale_data.iterrows():
        whale_table_data.append([
            str(row['Customer ID']),
            row['Email'],
            f"${row['Total Spend']:,.2f}",
            str(row['Order Count'])
        ])
    
    whale_table = Table(whale_table_data)
    whale_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(whale_table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("### 📂 Upload Options")
upload_mode = st.radio("Select Upload Mode", ["Single File Analysis", "Compare Multiple Files"], horizontal=True)

if upload_mode == "Single File Analysis":
    uploaded_file = st.file_uploader("Upload Shopify Orders CSV", type=['csv'])
else:
    uploaded_files = st.file_uploader("Upload Multiple Shopify Orders CSVs for Comparison", type=['csv'], accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) > 1:
        st.info(f"📊 {len(uploaded_files)} files uploaded for comparison")
    uploaded_file = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        required_columns = ['Order ID', 'Created at', 'Customer ID', 'Total', 'Email']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.info(f"Expected columns: {', '.join(required_columns)}")
        else:
            df = parse_dates(df, 'Created at')
            
            st.success(f"✅ Loaded {len(df)} orders from {df['Customer ID'].nunique()} unique customers")
            
            st.markdown("---")
            
            st.subheader("📅 Date Range Filter")
            df_clean_dates = df.dropna(subset=['Created at'])
            
            if len(df_clean_dates) > 0:
                min_date = df_clean_dates['Created at'].min().date()
                max_date = df_clean_dates['Created at'].max().date()
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
                
                df = df[(df['Created at'].dt.date >= start_date) & (df['Created at'].dt.date <= end_date)]
                
                if len(df) == 0:
                    st.warning("⚠️ No orders found in the selected date range. Please adjust your filters.")
                    st.stop()
                else:
                    st.info(f"📊 Analyzing {len(df)} orders from {df['Customer ID'].nunique()} customers in the selected period")
            
            st.markdown("---")
            
            st.header("📈 Monthly Cohort Retention Heatmap")
            st.markdown("Shows what percentage of customers return after their first purchase month")
            
            try:
                retention_matrix, cohort_size = calculate_cohort_retention(df)
                
                cohort_size_df = pd.DataFrame({
                    'Cohort Month': cohort_size.index.astype(str),
                    'Customers Acquired': cohort_size.values
                })
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(14, 8))
                    sns.heatmap(
                        retention_matrix,
                        annot=True,
                        fmt='.1f',
                        cmap='RdYlGn',
                        vmin=0,
                        vmax=100,
                        cbar_kws={'label': 'Retention %'},
                        ax=ax
                    )
                    ax.set_title('Monthly Cohort Retention (%)', fontsize=16, pad=20)
                    ax.set_xlabel('Months Since First Purchase', fontsize=12)
                    ax.set_ylabel('Cohort Month', fontsize=12)
                    
                    retention_matrix_str = retention_matrix.copy()
                    retention_matrix_str.index = retention_matrix_str.index.astype(str)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("👥 Cohort Sizes")
                    st.markdown("Customers acquired per month")
                    st.dataframe(
                        cohort_size_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    total_customers = cohort_size.sum()
                    avg_cohort_size = cohort_size.mean()
                    st.metric("Total Customers", f"{int(total_customers)}")
                    st.metric("Avg Cohort Size", f"{int(avg_cohort_size)}")
                
                with st.expander("📊 View Retention Data Table"):
                    st.dataframe(retention_matrix_str.style.format("{:.1f}%"))
                
                st.subheader("📥 Download Reports")
                col1, col2 = st.columns(2)
                
                with col1:
                    retention_csv = retention_matrix_str.to_csv()
                    st.download_button(
                        label="Download Retention Data (CSV)",
                        data=retention_csv,
                        file_name="cohort_retention.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    whale_customers_raw = get_whale_customers(df)
                    try:
                        pdf_buffer = create_pdf_report(retention_matrix, whale_customers_raw, cohort_size)
                        st.download_button(
                            label="Download Full Report (PDF)",
                            data=pdf_buffer,
                            file_name="shopify_analytics_report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as pdf_error:
                        st.error(f"PDF generation error: {str(pdf_error)}")
                
            except Exception as e:
                st.error(f"Error calculating cohort retention: {str(e)}")
            
            st.markdown("---")
            
            st.header("💰 Customer Lifetime Value (LTV) Analysis")
            
            try:
                ltv_metrics = calculate_customer_ltv(df)
                
                avg_ltv = ltv_metrics['LTV'].mean()
                median_ltv = ltv_metrics['LTV'].median()
                avg_order_value = ltv_metrics['Avg Order Value'].mean()
                avg_purchase_freq = ltv_metrics['Purchase Frequency'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg LTV (12 months)", f"${avg_ltv:,.2f}")
                with col2:
                    st.metric("Median LTV", f"${median_ltv:,.2f}")
                with col3:
                    st.metric("Avg Order Value", f"${avg_order_value:,.2f}")
                with col4:
                    st.metric("Avg Purchase Freq/Month", f"{avg_purchase_freq:.2f}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ltv_metrics['LTV'].hist(bins=20, edgecolor='black', ax=ax)
                    ax.set_title('Customer LTV Distribution', fontsize=14)
                    ax.set_xlabel('Lifetime Value ($)', fontsize=12)
                    ax.set_ylabel('Number of Customers', fontsize=12)
                    ax.axvline(avg_ltv, color='red', linestyle='--', linewidth=2, label=f'Mean: ${avg_ltv:,.2f}')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(ltv_metrics['Order Count'], ltv_metrics['Total Spend'], alpha=0.6)
                    ax.set_title('Order Count vs Total Spend', fontsize=14)
                    ax.set_xlabel('Number of Orders', fontsize=12)
                    ax.set_ylabel('Total Spend ($)', fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with st.expander("📊 View Top 20 Customers by LTV"):
                    top_ltv = ltv_metrics.nlargest(20, 'LTV')[['Customer ID', 'Total Spend', 'Order Count', 'Avg Order Value', 'LTV']]
                    top_ltv_display = top_ltv.copy()
                    top_ltv_display['Total Spend'] = top_ltv_display['Total Spend'].apply(lambda x: f"${x:,.2f}")
                    top_ltv_display['Avg Order Value'] = top_ltv_display['Avg Order Value'].apply(lambda x: f"${x:,.2f}")
                    top_ltv_display['LTV'] = top_ltv_display['LTV'].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(top_ltv_display.reset_index(drop=True), use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error calculating LTV: {str(e)}")
            
            st.markdown("---")
            
            st.header("🐋 Whale Customers (Top 10 by Total Spend)")
            
            try:
                whale_customers = get_whale_customers(df)
                
                whale_customers_display = whale_customers.copy()
                whale_customers_display['Total Spend'] = whale_customers_display['Total Spend'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(
                    whale_customers_display.reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
                
                total_whale_spend = df[df['Customer ID'].isin(whale_customers['Customer ID'])]['Total'].sum()
                total_revenue = df['Total'].sum()
                whale_percentage = (total_whale_spend / total_revenue) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Top 10 Customers Spend", f"${total_whale_spend:,.2f}")
                with col2:
                    st.metric("Total Revenue", f"${total_revenue:,.2f}")
                with col3:
                    st.metric("Whale Contribution", f"{whale_percentage:.1f}%")
                
                whale_csv = whale_customers.to_csv(index=False)
                st.download_button(
                    label="Download Whale Customers (CSV)",
                    data=whale_csv,
                    file_name="whale_customers.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error calculating whale customers: {str(e)}")
                
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.info("Please ensure your CSV has the correct format with columns: Order ID, Created at, Customer ID, Total, Email")
elif upload_mode == "Compare Multiple Files" and 'uploaded_files' in locals() and uploaded_files and len(uploaded_files) >= 2:
    st.header("📊 Multi-File Comparison View")
    
    try:
        comparison_data = []
        
        for idx, file in enumerate(uploaded_files):
            df_temp = pd.read_csv(file)
            required_columns = ['Order ID', 'Created at', 'Customer ID', 'Total', 'Email']
            missing_columns = [col for col in required_columns if col not in df_temp.columns]
            
            if not missing_columns:
                df_temp = parse_dates(df_temp, 'Created at')
                
                comparison_data.append({
                    'File Name': file.name,
                    'Total Orders': len(df_temp),
                    'Unique Customers': df_temp['Customer ID'].nunique(),
                    'Total Revenue': df_temp['Total'].sum(),
                    'Avg Order Value': df_temp['Total'].mean(),
                    'Date Range Start': df_temp['Created at'].min(),
                    'Date Range End': df_temp['Created at'].max()
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            st.subheader("📋 Summary Comparison")
            
            comparison_display = comparison_df.copy()
            comparison_display['Total Revenue'] = comparison_display['Total Revenue'].apply(lambda x: f"${x:,.2f}")
            comparison_display['Avg Order Value'] = comparison_display['Avg Order Value'].apply(lambda x: f"${x:,.2f}")
            comparison_display['Date Range Start'] = comparison_display['Date Range Start'].dt.strftime('%Y-%m-%d')
            comparison_display['Date Range End'] = comparison_display['Date Range End'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(comparison_display, use_container_width=True, hide_index=True)
            
            st.subheader("📈 Visual Comparisons")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(comparison_df['File Name'], comparison_df['Total Revenue'])
                ax.set_title('Total Revenue Comparison', fontsize=14)
                ax.set_xlabel('File', fontsize=12)
                ax.set_ylabel('Revenue ($)', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(comparison_df['File Name'], comparison_df['Unique Customers'])
                ax.set_title('Unique Customers Comparison', fontsize=14)
                ax.set_xlabel('File', fontsize=12)
                ax.set_ylabel('Customers', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            st.download_button(
                label="Download Comparison Data (CSV)",
                data=comparison_df.to_csv(index=False),
                file_name="file_comparison.csv",
                mime="text/csv"
            )
        else:
            st.warning("Unable to process uploaded files. Please check the format.")
            
    except Exception as e:
        st.error(f"Error processing files for comparison: {str(e)}")
        
else:
    st.info("👆 Upload a Shopify orders CSV file to begin analysis")
    
    st.markdown("### Expected CSV Format")
    st.code("""Order ID,Created at,Customer ID,Total,Email
1001,2024-01-03,201,129.5,john.doe@example.com
1002,2024-01-05,202,89.0,jane.smith@example.com""")
