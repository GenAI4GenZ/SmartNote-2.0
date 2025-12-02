import streamlit as st
import re
from docker_utils import execute_docker_command, test_docker_setup
from demo_config import REPO_LIST, GITHUB_TOKEN, OPENAI_API_KEY


def extract_release_note(output: str) -> str:
    """
    Extract the release note content from the docker command output.
    The release note appears after the LAST line containing:
    'DEBUG | smartnote.prompts_manager:send_request:91 - <OpenAI> response:'
    """
    lines = output.split('\n')
    
    # Find the LAST occurrence of the OpenAI response line that contains the final release note
    final_response_idx = -1
    for i, line in enumerate(lines):
        if "smartnote.prompts_manager:send_request:91 - <OpenAI> response:" in line:
            final_response_idx = i
            # Don't break - we want the LAST occurrence
    
    # If we found the marker, start extracting from the next line
    if final_response_idx == -1:
        return ""
    
    # Start from the line after the OpenAI response marker
    start_idx = final_response_idx + 1
    
    # Skip any empty lines at the start
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1
    
    # Extract until we hit the next timestamp log line or end of output
    release_note_lines = []
    for i in range(start_idx, len(lines)):
        line = lines[i]
        # Stop if we hit a timestamp log line (these come after the release note)
        if re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', line):
            break
        # Stop if we hit a command prompt
        if re.match(r'^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+:', line):
            break
        release_note_lines.append(line)
    
    return '\n'.join(release_note_lines).strip()


st.set_page_config(page_title="Release Note Generator", page_icon="📝", layout="wide")

st.title("📝 Release Note Generator")
st.markdown("Generate release notes for GitHub repositories using SmartNote")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Test Docker setup
    docker_ok, docker_msg = test_docker_setup()
    if docker_ok:
        st.success("✅ Docker is ready")
    else:
        st.error(f"❌ {docker_msg}")
    
    st.info("🐳 Using SmartNote Docker image locally")
    st.info("📦 Image: ghcr.io/genai4genz/smartnote:latest")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Parameters")
    
    # Repo selection
    selected_repo = st.selectbox(
        "Select Repository",
        options=REPO_LIST,
        help="Choose a GitHub repository in format: owner/repo"
    )
    
    # Version inputs
    previous_version = st.text_input(
        "Previous Release Version",
        placeholder="e.g., 1.9.2 or v1.9.2",
        help="Enter the previous release version (with or without 'v' prefix)"
    )
    
    current_version = st.text_input(
        "Current Release Version",
        placeholder="e.g., 1.9.3 or v1.9.3",
        help="Enter the current release version (with or without 'v' prefix)"
    )
    
    # Additional options
    with st.expander("Advanced Options"):
        group_commits = st.checkbox("Group Commits", value=True)
        show_significance = st.checkbox("Show Significance", value=True)
        use_gpu = st.checkbox("Use GPU (if available)", value=True)
    
    # Generate button
    generate_button = st.button("🚀 Generate Release Note", type="primary", use_container_width=True)

with col2:
    st.subheader("Release Note Output")
    
    if generate_button:
        if not selected_repo:
            st.error("Please select a repository")
        elif not previous_version or not current_version:
            st.error("Please enter both previous and current release versions")
        else:
            # Use version as provided by user (don't add 'v' prefix)
            prev_ver = previous_version.strip()
            curr_ver = current_version.strip()
            
            with st.spinner("Generating release note... This may take a few minutes."):
                try:
                    # Execute Docker command locally
                    success, output = execute_docker_command(
                        repo_name=selected_repo,
                        previous_release=prev_ver,
                        current_release=curr_ver,
                        github_token=GITHUB_TOKEN,
                        openai_api_key=OPENAI_API_KEY,
                        group_commits=group_commits,
                        show_significance=show_significance,
                        use_gpu=use_gpu
                    )
                    
                    if success:
                        # Extract release note from output
                        release_note = extract_release_note(output)
                        
                        if release_note:
                            st.success("✅ Release note generated successfully!")
                            
                            # Display in markdown
                            st.markdown(release_note)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Release Note",
                                data=release_note,
                                file_name=f"release_note_{selected_repo.replace('/', '_')}_{current_version}.md",
                                mime="text/markdown"
                            )
                        else:
                            st.warning("Release note was generated but couldn't be extracted from output")
                            with st.expander("View Full Output"):
                                st.text(output)
                    else:
                        st.error("❌ Failed to generate release note")
                        with st.expander("View Error Details"):
                            st.text(output)
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
    else:
        st.info("👈 Configure parameters and click 'Generate Release Note' to start")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Powered by SmartNote • GitHub Release Note Generator
    </div>
    """,
    unsafe_allow_html=True
)
