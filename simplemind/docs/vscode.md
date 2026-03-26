# Connecting to a server via SSH

An SSH (Secure Shell) connection enables two computers to communicate and is how you will connect to a GPU server from your desktop or laptop:

1. Open VS Code and on the left menu bar, select the *extensions* icon and install the *Remote Development* package.
    - This package allows you to open remote folders on a remote server using SSH.
2. Generate an SSH key (if necessary)
    - Check to see if you already have an SSH key on your **local** computer. 
        - For MacOS and Linux this is located at ~/.ssh/id_rsa.pub
        - For windows this is in the .ssh directory in your user profile folder (for example C:\Users\your-user\.ssh\id_rsa.pub)
    - If you do not have a key, run the following command in a **local** terminal (MacOS & Linux)/PowerShell (Windows) to generate an SSH key pair: `ssh-keygen -t rsa -b 4096`
        - Do not enter anything for the prompts that follow (Enter file in which to save the key, Enter passphrase, Enter same passphrase again), just hit enter/return.
3. Setup the key-based authentication for Linux servers.
    - Locate the newly generated id_rsa.pub file on local machine, open it in Notepad, and copy its contents.
    - SSH into Linux server:
        - To do this, go to *Open a Remote Window* on VSCode (bottom left, in blue), hit *Connect to a Host* > *Add new SSH Host*.
        - Use `username@hostname` to log in.
        - **The server hostname will be provided for PBMED 210.**
    - Run command `ls -a` and look for a folder named `.ssh`
    - If it doesn’t exist, run command `mkdir .ssh`
    - Open authorized key file in target server: `vim ~/.ssh/authorized_keys`
    - Copy the contents of the *id_ras.pub* file and paste with the following steps:
        - If the file is not empty, navigate to the end of the file with the arrow keys or *End* key.
        - Hit *a* to change to append mode.
        - Use *Ctrl+C* to paste the contents of the *id_rsa.pub* file.
        - Hit *ESC** to exit append mode.
        - Type `:wq` to save and exit.
    - Run command `chmod 600 ~/.ssh/authorized_keys` in the terminal command line to ensure the file permissions are set correctly.
    - You should now be able to SSH to a target remote server from your local computer without typing your password.
        - Close the remote terminal window and try again to SSH back to the target server.
        - If you are prompted for your password, something went wrong.

Next, set the shell startup script:
* Open Command Palette (Ctrl+Shift+P)
* Type:
``` bash
Preferences: Open Remote Settings (JSON)
```

* This opens a settings.json that applies only to this remote server, paste:
``` json
{
    "terminal.integrated.shellIntegration.enabled": false,
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.profiles.linux": {
        "bash": {
            "path": "/usr/bin/bash",
            "args": ["-l"]
        }
    }
}
```

* Open a new bash terminal.
