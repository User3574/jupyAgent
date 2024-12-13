import gradio as gr

def combine_inputs(title, content):
    # Create a simple HTML template with the inputs
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #2c3e50;">{title}</h1>
        <div style="background-color: #f7f9fc; padding: 15px; border-radius: 8px;">
            <p style="line-height: 1.6;">{content}</p>
        </div>
        <footer style="margin-top: 20px; color: #7f8c8d; font-size: 0.9em;">
            Generated with Gradio
        </footer>
    </div>
    """
    return html

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# HTML Generator")
    
    with gr.Row():
        title_input = gr.Textbox(label="Title", placeholder="Enter your title here")
        content_input = gr.Textbox(label="Content", placeholder="Enter your content here", lines=3)
    
    generate_btn = gr.Button("Generate HTML")
    output = gr.HTML(label="Generated HTML")
    
    generate_btn.click(
        fn=combine_inputs,
        inputs=[title_input, content_input],
        outputs=output
    )

demo.launch()