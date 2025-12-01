import subprocess
import time
from typing import Tuple


def execute_docker_command(
    repo_name: str,
    previous_release: str,
    current_release: str,
    github_token: str,
    openai_api_key: str,
    group_commits: bool = True,
    show_significance: bool = True,
    use_gpu: bool = True,
    timeout: int = 600
) -> Tuple[bool, str]:
    """
    Execute the SmartNote docker command locally.
    
    Args:
        repo_name: GitHub repository in format owner/repo
        previous_release: Previous release version (e.g., v1.37.3)
        current_release: Current release version (e.g., v1.37.4)
        github_token: GitHub API token
        openai_api_key: OpenAI API key
        group_commits: Whether to group commits
        show_significance: Whether to show significance scores
        use_gpu: Whether to use GPU (will fallback if not available)
        timeout: Command timeout in seconds
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    # Build the docker command
    docker_cmd = build_docker_command(
        repo_name=repo_name,
        previous_release=previous_release,
        current_release=current_release,
        github_token=github_token,
        openai_api_key=openai_api_key,
        group_commits=group_commits,
        show_significance=show_significance,
        use_gpu=use_gpu
    )
    
    try:
        # Execute Docker command locally
        process = subprocess.Popen(
            docker_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Collect output
        output_lines = []
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line.rstrip())
            
            # Wait for process to complete
            return_code = process.wait(timeout=timeout)
            
        except subprocess.TimeoutExpired:
            process.kill()
            return False, f"Command timed out after {timeout} seconds."
        
        full_output = '\n'.join(output_lines)
        
        # Check if command was successful
        success = return_code == 0 or "Release note generated" in full_output
        
        # If GPU failed, try again with CPU
        if not success and use_gpu and "CUDA" in full_output:
            return execute_docker_command(
                repo_name=repo_name,
                previous_release=previous_release,
                current_release=current_release,
                github_token=github_token,
                openai_api_key=openai_api_key,
                group_commits=group_commits,
                show_significance=show_significance,
                use_gpu=False,
                timeout=timeout
            )
        
        return success, full_output
        
    except Exception as e:
        return False, f"Error executing Docker command: {str(e)}"


def build_docker_command(
    repo_name: str,
    previous_release: str,
    current_release: str,
    github_token: str,
    openai_api_key: str,
    group_commits: bool = True,
    show_significance: bool = True,
    use_gpu: bool = True
) -> str:
    """
    Build the docker command to run SmartNote.
    
    Returns:
        Complete docker command string
    """
    base_cmd = "docker run"
    
    # Add GPU flag if requested
    if use_gpu:
        base_cmd += " --gpus all"
    
    # Add standard flags (no -it for automated execution)
    base_cmd += " --rm"
    
    # Add environment variables
    base_cmd += f' -e SMARTNOTE_GITHUB__TOKEN="{github_token}"'
    base_cmd += f' -e SMARTNOTE_OPENAI__API_KEY="{openai_api_key}"'
    
    # Add docker image from SmartNote repository
    base_cmd += " ghcr.io/genai4genz/smartnote:latest"
    
    # Add repository
    base_cmd += f" {repo_name}"
    
    # Add release versions
    base_cmd += f" --previous-release {previous_release}"
    base_cmd += f" --current-release {current_release}"
    
    # Add optional flags
    if group_commits:
        base_cmd += " --group-commits"
    if show_significance:
        base_cmd += " --show-significance"
    
    return base_cmd


def test_docker_setup() -> Tuple[bool, str]:
    """
    Test if Docker is installed and accessible.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return True, f"Docker is available: {result.stdout.strip()}"
        else:
            return False, "Docker is installed but not responding correctly"
            
    except FileNotFoundError:
        return False, "Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
    except Exception as e:
        return False, f"Error checking Docker: {str(e)}"


def pull_smartnote_image() -> Tuple[bool, str]:
    """
    Pull the SmartNote Docker image.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        result = subprocess.run(
            ["docker", "pull", "ghcr.io/genai4genz/smartnote:latest"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return True, "SmartNote image pulled successfully"
        else:
            return False, f"Failed to pull image: {result.stderr}"
            
    except Exception as e:
        return False, f"Error pulling image: {str(e)}"
