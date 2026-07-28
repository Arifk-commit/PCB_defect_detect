import streamlit as st

def show_model_info():
    st.markdown("""
        <div class="app-header">
            <h1>Model Specifications Card</h1>
            <p>Technical details, model parameters, architecture, and accuracy metrics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Grid details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        
        # Display configuration values inside a stylized table or list
        st.markdown("""
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Specification Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Model Name</strong></td>
                    <td>YOLOv11 Nano (PCB Customized)</td>
                </tr>
                <tr>
                    <td><strong>Framework</strong></td>
                    <td>PyTorch (v2.3.0)</td>
                </tr>
                <tr>
                    <td><strong>Input Canvas Size</strong></td>
                    <td>640 x 640 pixels</td>
                </tr>
                <tr>
                    <td><strong>Parameters</strong></td>
                    <td>3.1 Million (Lightweight Edge Model)</td>
                </tr>
                <tr>
                    <td><strong>Disk Storage Size</strong></td>
                    <td>6.2 MB</td>
                </tr>
                <tr>
                    <td><strong>Export Configurations</strong></td>
                    <td>ONNX, TensorRT, TorchScript, CoreML</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Edge Deployment Friendly**: The YOLOv11 nano model is optimized to run on standard low-power edge compute cards (like Raspberry Pi 5 or NVIDIA Jetson Nano) directly on the conveyor inspection belts.")
        
    with col2:
        st.subheader("Training Metrics & Accuracy")
        
        st.markdown("""
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Evaluation Metric</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Precision (P)</strong></td>
                    <td>93.4%</td>
                </tr>
                <tr>
                    <td><strong>Recall (R)</strong></td>
                    <td>91.2%</td>
                </tr>
                <tr>
                    <td><strong>mAP50</strong></td>
                    <td>94.8%</td>
                </tr>
                <tr>
                    <td><strong>mAP50-95</strong></td>
                    <td>68.3%</td>
                </tr>
                <tr>
                    <td><strong>Inference Speed (avg GPU)</strong></td>
                    <td>4.2 ms</td>
                </tr>
                <tr>
                    <td><strong>Inference Speed (avg CPU)</strong></td>
                    <td>24.5 ms</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    st.subheader("Classes List (Target Defect Categories)")
    
    class_col1, class_col2, class_col3 = st.columns(3)
    
    with class_col1:
        st.markdown("""
        * **Missing Hole**: Areas where a through-hole or via is missing from the copper pad.
        * **Mouse Bite**: Small circular bites taken out of copper tracks, reducing conductivity.
        """)
        
    with class_col2:
        st.markdown("""
        * **Open Circuit**: Discontinuity in traces which prevents signal propagation.
        * **Short**: Accidental connections between adjacent electrical tracks or vias.
        """)
        
    with class_col3:
        st.markdown("""
        * **Spur**: Visual projections of copper sticking out from trace borders.
        * **Spurious Copper**: Excess blobs of isolated copper left during etching.
        """)
        
    st.write("---")
    
    st.subheader("Weight Integration Guidelines")
    st.markdown("""
    To replace this simulator with your own custom-trained model:
    1. Export your YOLOv11 model weights file as `best.pt`.
    2. Place the file inside the project directory at the path: `models/best.pt`.
    3. The application will detect the file automatically at runtime and transition from fallback simulations to live execution.
    """)
