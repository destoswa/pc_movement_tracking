import subprocess
from pathlib import Path

def run_icp(source, target, output, exe_path):
    """Run ICP using PCL executable."""

    cmd = [
        str(exe_path),
        str(source),
        str(target),
        str(output),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError("ICP failed")

    return output


# Example usage
run_icp(
    "data/test_pcl/2500_1144_2018.ply",
    "data/test_pcl/2500_1144_2024.ply",
    "data/test_pcl/aligned.ply",
    "build/Release/icp.exe"
)