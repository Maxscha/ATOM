mkdir tmp
mkdir ast_process/data
curl https://zenodo.org/record/4077754/files/functions_extracted_commits.tar.gz?download=1 -o tmp/functions_extracted_commits.tar.gz
curl https://zenodo.org/record/4077754/files/functions_filter_commits.tar.gz?download=1 -o tmp/functions_filter_commits.tar.gz
#curl https://zenodo.org/record/4077754/files/raw_commits_from_github.tar.gz?download=1 -o tmp/raw_commits_from_github.tar.gz

tar -xf tmp/functions_extracted_commits.tar.gz -C ast_process/data/
tar -xf tmp/functions_filter_commits.tar.gz -C ast_process/data/
tar -xf tmp/raw_commits_from_github.tar.gz -C ast_process/data/

# cp -r tmp/functions_extracted_commits ast_process/data/functions_extracted_commits
# cp -r tmp/raw_commits_from_github ast_process/data/raw_commits_from_github

python3 retrieve_repo_urls.py > ast_process/data/repo_urls.txt

mkdir ast_process/data/repos
cd ast_process/data/repos
### Clone all repos in directoy but maybe we only start with one to test the pipeline
while read repo; do
    git clone "$repo" 
done < ../repo_urls.txt
cd ../../..
