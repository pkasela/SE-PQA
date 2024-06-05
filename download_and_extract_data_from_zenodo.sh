# list of all urls to download
declare -a UrlArray=(
    "https://zenodo.org/records/10679181/files/answers.csv"
    "https://zenodo.org/records/10679181/files/comments.csv"
    "https://zenodo.org/records/10679181/files/postlinks.csv"
    "https://zenodo.org/records/10679181/files/questions.csv"
    "https://zenodo.org/records/10679181/files/questions_with_answer.csv"
    "https://zenodo.org/records/10679181/files/SE-PQA.zip"
    "https://zenodo.org/records/10679181/files/tags.csv"
    "https://zenodo.org/records/10679181/files/users.csv"




)

mkdir -p PIRFIRE24_data
cd PIRFIRE24_data
# dowload files

for url in "${UrlArray[@]}"; do
    url_file=${url##*/}
    if [ -f $url_file ]; then
        echo "$url_file exists."
    else
        wget --tries=0 $url
    fi
done


# unzip all files
unzip SE-PQA.zip

        
done

cd ..