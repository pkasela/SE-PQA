# list of all urls to download
declare -a UrlArray=(
    "https://archive.org/download/stackexchange/writers.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/workplace.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/woodworking.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/vegetarianism.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/travel.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/sustainability.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/sports.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/sound.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/skeptics.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/scifi.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/rpg.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/politics.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/philosophy.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/pets.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/parenting.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/outdoors.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/opensource.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/musicfans.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/music.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/movies.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/money.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/martialarts.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/literature.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/linguistics.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/lifehacks.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/law.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/judaism.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/islam.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/interpersonal.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/hsm.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/history.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/hinduism.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/hermeneutics.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/health.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/genealogy.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/gardening.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/gaming.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/freelancing.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/fitness.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/expatriates.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/english.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/diy.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/cooking.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/christianity.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/buddhism.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/boardgames.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/bicycles.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/apple.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/anime.stackexchange.com.7z"
    "https://archive.org/download/stackexchange/academia.stackexchange.com.7z"
)

mkdir -p raw_data
cd raw_data
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
for url in "${UrlArray[@]}"; do
    url_file=${url##*/}
    domain=${url_file%%.*}
    py7zr x $url_file ./$domain
done

cd ..