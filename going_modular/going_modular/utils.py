
"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

import os
import re
from pathlib import Path
from typing import List, Union

def get_sorted_checkpoints(
    model_dir: Union[str, Path],
    extension: str = ".pth",
    verbose: bool = True
) -> List[Path]:
    """
    Finds and returns all model checkpoint files with datetime in their name,
    sorted from newest to oldest based on the embedded timestamp.

    Args:
        model_dir (str or Path): Directory containing model checkpoints.
        extension (str): File extension to look for (default: ".pth").
        verbose (bool): If True, print summary info.

    Returns:
        List[Path]: List of Path objects sorted by timestamp (newest first).
    """
    model_dir = Path(model_dir)
    datetime_pattern = re.compile(r"\d{8}-\d{6}")

    # Get all files with given extension
    pth_files = list(model_dir.glob(f"*{extension}"))

    # Keep only files with a timestamp
    sorted_pth_files = sorted(
        [p for p in pth_files if datetime_pattern.search(p.stem)],
        key=lambda p: datetime_pattern.search(p.stem).group(),
        reverse=True
    )

    if verbose:
        if sorted_pth_files:
            print(f"🕒 Latest checkpoint:\n{sorted_pth_files[0]}\n")
            print("📁 All sorted checkpoint files:")
            for f in sorted_pth_files:
                print(str(f))
        else:
            print("⚠️ No valid checkpoint files with timestamp found.")

    return sorted_pth_files

import smtplib
from email.mime.text import MIMEText

def send_email_notification(
    subject: str,
    body: str,
    to_email: str,
    from_email: str,
    app_password: str,
    quit_runtime: bool = True
):
    """
    Sends an email notification using Gmail's SMTP.

    Args:
        subject (str): Subject of the email.
        body (str): Body text of the email.
        to_email (str): Recipient email address.
        from_email (str): Sender Gmail address.
        app_password (str): App-specific Gmail password.
        quit_runtime (bool): Whether to shut down the Colab runtime after sending.
    """
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(from_email, app_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print('📬 Email notification sent successfully.')
    except Exception as e:
        print(f'❌ Failed to send email: {e}')

    if quit_runtime:
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as e:
            print(f'⚠️ Failed to quit Colab runtime: {e}')


