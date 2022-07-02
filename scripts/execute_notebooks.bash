


NOTEBOOKS=$(find . -type f -name "*.ipynb" -not -path '*/.*')
echo $NOTEBOOKS

for file in $NOTEBOOKS
do
    echo "Executing $file"
#    jupyter nbconvert --to notebook --execute $file
done
