black slam_train/

isort slam_train/

printf "\nPress any key to continue to pylint...\n"
read -n 1 -s -r
pylint slam_train/

printf "\nPress any key to continue to mypy...\n"
read -n 1 -s -r
mypy slam_train/
