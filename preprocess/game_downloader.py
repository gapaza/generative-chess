import os
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import config
from concurrent.futures import ThreadPoolExecutor


# """
# wget -r -l inf -np -nH --cut-dirs=3 -A .pgn  --no-check-certificate -e robots=off https://storage.lczero.org/files/match_pgns/3/
# """


# lc0_folder = os.path.join(config.games_dir, 'lc0')
# lc0_folder_pgn = os.path.join(lc0_folder, 'pgn3')
# if not os.path.exists(lc0_folder):
#     os.makedirs(lc0_folder)
# if not os.path.exists(lc0_folder_pgn):
#     os.makedirs(lc0_folder_pgn)

lc0_folder_pgn = '/home/ubuntu/games/lc0/tar'


def list_files(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a')
        # ending = '.pgn'
        ending = '.tar.bz2'
        files = [link.get('href') for link in links if link.get('href').endswith(ending)]
        return files
    else:
        print("Failed to retrieve the directory listing")
        return []

def download_file(url, destination_folder, filename):
    path = f"{destination_folder}/{filename}"
    if os.path.exists(path):
        print(f"File {filename} already exists. Skipping download.")
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")


def download_url_files(url, destination_folder):
    pgn_files = list_files(url)
    # pgn_files = pgn_files[:100]
    num_threads = 10
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(download_file, url + file_name, destination_folder, file_name) for file_name in pgn_files]
        for future in futures:
            future.result()  # Wait for all futures to complete, handling them as they complete.





if __name__ == '__main__':
    destination_folder = lc0_folder_pgn
    # url = 'https://storage.lczero.org/files/match_pgns/3/'  # 35852 pgn files
    url = 'https://storage.lczero.org/files/training_pgns/test80/'
    download_url_files(url, destination_folder)








"""

tar --use-compress-program=pbzip2 -xf pgns-run1-test80-20220404-0654.tar.bz2 -C /home/ubuntu/games/lc0/set


tar --use-compress-program=pbzip2 -xf pgns-run1-test80-20220405-1854.tar.bz2 -C /home/ubuntu/games/lc0/set



for file in *.tar.bz2; do
  tar --use-compress-program=pbzip2 -xf "$file" -C /home/ubuntu/games/lc0/set
done


"""




