# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Drowning Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Drowning-probability detection")

# Sidebar
st.sidebar.header("ML Model Config")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 15, 60, 25)) / 100

model_path = 'best.pt'
# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    if source_img is None:
        default_image_path = str(settings.DEFAULT_IMAGE)
        default_image = PIL.Image.open(default_image_path)
        st.image(default_image_path, caption="Default Image",
                    use_column_width=True)
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img, caption="Uploaded Image",
                    use_column_width=True)
        
    if source_img is None:
        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
        default_detected_image = PIL.Image.open(
            default_detected_image_path)
        st.image(default_detected_image_path, caption='Detected Probability',
                    use_column_width=True)
    else:
        if st.sidebar.button('Detect Drowning'):
            res = model.predict(uploaded_image,
                                conf=confidence
                                )
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                        use_column_width=True)
            try:
                with st.expander("Detected Probability"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

settings.account_sid = st.sidebar.text_input("Your Twilio account_sid")
settings.auth_token = st.sidebar.text_input("Your Twilio auth_token")
settings.to_ = st.sidebar.text_input("Your Twilio to_")
settings.from_ = st.sidebar.text_input("Your Twilio from_")
settings.imgbb_api = st.sidebar.text_input("Your ImgBB api-key")

# Data Visualization
st.sidebar.header("Data Visualization")

if st.sidebar.checkbox('Show Data Visualization'):

    # Example data for detection alerts
    alerts_data = {
        'date': pd.date_range(start='2023-01-01', periods=30),
        'alerts': [5, 10, 7, 12, 8, 6, 9, 15, 10, 5, 8, 11, 6, 7, 12, 9, 10, 8, 11, 13, 7, 9, 12, 8, 10, 11, 9, 7, 10, 8]
    }
    alerts_df = pd.DataFrame(alerts_data)

    # Example data for swimmer activity
    swimmer_activity_data = {
        'hour': np.tile(np.arange(24), 10),
        'swimmers': np.random.randint(0, 20, size=240)
    }
    swimmer_activity_df = pd.DataFrame(swimmer_activity_data)

    # Example data for system performance
    system_performance_data = {
        'time': pd.date_range(start='2023-01-01', periods=24, freq='H'),
        'inference_time': np.random.uniform(0.1, 0.3, size=24),
        'gpu_usage': np.random.uniform(30, 80, size=24),
        'cpu_usage': np.random.uniform(20, 70, size=24)
    }
    system_performance_df = pd.DataFrame(system_performance_data)

    # Example data for response metrics
    response_metrics_data = {
        'response_time': np.random.normal(loc=2, scale=0.5, size=50)
    }
    response_metrics_df = pd.DataFrame(response_metrics_data)

    # Detection Alerts Over Time - Line Chart
    alerts_line_chart = px.line(alerts_df, x='date', y='alerts', title='Drowning Alerts Over Time')
    st.plotly_chart(alerts_line_chart)

    # Detection Alerts Over Time - Bar Chart (Weekly)
    alerts_df['week'] = alerts_df['date'].dt.isocalendar().week
    weekly_alerts_df = alerts_df.groupby('week')['alerts'].sum().reset_index()
    alerts_bar_chart = px.bar(weekly_alerts_df, x='week', y='alerts', title='Weekly Drowning Alerts')
    st.plotly_chart(alerts_bar_chart)

    # Swimmer Activity - Heatmap
    heatmap_data = np.random.randint(0, 20, size=(10, 10))  # Example heatmap data
    heatmap = go.Figure(data=go.Heatmap(z=heatmap_data))
    heatmap.update_layout(title='Heatmap of Swimmer Locations')
    st.plotly_chart(heatmap)

    # Swimmer Activity - Line Chart
    swimmer_line_chart = px.line(swimmer_activity_df, x='hour', y='swimmers', title='Number of Swimmers Detected Per Hour')
    st.plotly_chart(swimmer_line_chart)

    # System Performance - Line Chart for Inference Time
    inference_time_line_chart = px.line(system_performance_df, x='time', y='inference_time', title='Inference Time Per Frame Over the Last 24 Hours')
    st.plotly_chart(inference_time_line_chart)

    # System Performance - Bar Chart for GPU/CPU Usage
    gpu_cpu_usage_bar_chart = go.Figure(data=[
        go.Bar(name='GPU Usage', x=system_performance_df['time'], y=system_performance_df['gpu_usage']),
        go.Bar(name='CPU Usage', x=system_performance_df['time'], y=system_performance_df['cpu_usage'])
    ])
    gpu_cpu_usage_bar_chart.update_layout(barmode='group', title='GPU/CPU Usage Over the Last 24 Hours')
    st.plotly_chart(gpu_cpu_usage_bar_chart)

    # Response Metrics - Histogram
    response_time_histogram = px.histogram(response_metrics_df, x='response_time', nbins=10, title='Response Times to Alerts')
    st.plotly_chart(response_time_histogram)

    # Response Metrics - Box Plot
    response_time_box_plot = px.box(response_metrics_df, y='response_time', title='Distribution of Response Times')
    st.plotly_chart(response_time_box_plot)

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p><a style='display: block; text-align: center;' 
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
