#!/bin/bash

# Configuration
REPO_DIR="/workspace/slice-monorepo"  # Replace this with the path to your local repository
BRANCH="cl_pipeline_runpod"
SSH_DIR="$HOME/.ssh"
SSH_KEY_NAME="runpod"
SSH_PRIVATE_KEY_PATH="$SSH_DIR/$SSH_KEY_NAME"
GIT_USER_NAME="charles-sliced"            # Set your name here
GIT_USER_EMAIL="charles@sliced-ai.com"     # Set your email here

# Ensure the SSH agent is running and the key is added
eval "$(ssh-agent -s)"
ssh-add "$SSH_PRIVATE_KEY_PATH"

# Navigate to the repository directory
cd "$REPO_DIR" || exit

# Configure git user and email if not already set
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

# Remove existing .gitignore and recreate it
rm -f .gitignore
echo -e "*.csv\n*.json\n*.pyc\n*.npy\n*.png\n*.h5\n.ipynb_checkpoints" > .gitignore
find . -size +50M | sed 's|^\./||' >> .gitignore

# Create a pre-commit hook to prevent committing large files
HOOK_DIR="$REPO_DIR/.git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

cat > "$HOOK_FILE" << 'EOF'
#!/bin/sh
# Prevent large files from being committed
maxsize=50000000
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ] && [ $(wc -c <"$file") -gt $maxsize ]; then
        echo "Error: Attempt to commit file larger than 50MB: $file"
        exit 1
    fi
done
EOF

chmod +x "$HOOK_FILE"

# Display the current status
echo "Current status of the repository:"
git status

# Ask for user confirmation to proceed with adding all changes
read -p "Do you want to add all changes to staging? (y/n): " add_confirm
if [[ $add_confirm == "y" || $add_confirm == "Y" ]]; then
    git add .
    echo "All changes added to staging."
else
    echo "Exiting without adding changes."
    exit 1
fi

# Commit the changes
read -p "Enter your commit message: " commit_message
git commit -m "$commit_message"
if [ $? -eq 0 ]; then
    echo "Changes committed with message: $commit_message"
else
    echo "Commit failed due to large files."
    exit 1
fi

# Pull latest changes from the remote branch
echo "Pulling latest changes from the remote branch..."
git pull origin "$BRANCH"

# Push the changes
echo "Pushing changes to remote..."
git push origin "$BRANCH"
echo "Changes pushed to $BRANCH successfully."

# Kill the SSH agent
eval "$(ssh-agent -k)"

# End of script
echo "Script completed."
