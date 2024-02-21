release-patch commit-message='': # Default value is an empty string
    # Set error handling
    set -e

    # Increment the patch version and get the new version
    NEW_VERSION=$(poetry version -s)

    # Format the code and run linters
    poetry run black src/
    poetry run ruff src/

    # Install dependencies and run tests
    poetry install
    poetry run pytest

    # Build the package
    poetry build

    # Commit, tag, and publish
    git add .
    git commit --no-verify -m "${commit-message:-Version bump} - Version $(poetry version -s)"
    git push
    git tag -a "$NEW_VERSION" -m "Version $NEW_VERSION"
    git push origin --tags
    poetry publish
