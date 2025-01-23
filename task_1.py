import logging
from colorama import init, Fore
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt

# ==========================
# Logger & colorama setup
# ==========================
init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


class WordCountMapReduce:
    """
    Class that downloads text from a URL, performs a parallel (MapReduce) word-count,
    and visualizes the top N words.
    """

    def __init__(
        self, url: str, chunk_size: int = 2000, max_workers: int = 4, top_n: int = 10
    ):
        """
        :param url: URL to fetch text from
        :param chunk_size: number of words per chunk for parallel map step
        :param max_workers: number of threads for parallel execution
        :param top_n: how many top words to show in visualization
        """
        self.url = url
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.top_n = top_n
        self.words = []
        self.word_counts = {}

    def fetch_text(self) -> str:
        """
        Downloads text data from the URL and returns it as a string.
        """
        logger.info(Fore.BLUE + f"Fetching text from: {self.url}")
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.text
        except requests.RequestException as e:
            logger.error(Fore.RED + f"Failed to fetch from URL. Error: {e}")
            return ""

    def text_clean_and_split(self, text: str) -> None:
        """
        Cleans and splits the text into words. Updates self.words list.
        """
        logger.info(Fore.GREEN + "Cleaning and splitting text into words...")
        # Залишимо лише a-z, A-Z. Замінимо все інше на пробіли.
        cleaned_text = re.sub(r"[^a-zA-Z]+", " ", text)
        # Приведемо до нижнього регістру
        cleaned_text = cleaned_text.lower()
        # Розіб'ємо на слова
        self.words = cleaned_text.split()
        logger.info(Fore.GREEN + f"Total words extracted: {len(self.words)}")

    @staticmethod
    def mapper(chunk: list) -> dict:
        """
        Mapper function: takes a list of words (chunk) and returns a dict {word: count}.
        """
        local_count = defaultdict(int)
        for word in chunk:
            local_count[word] += 1
        return local_count

    @staticmethod
    def reducer(dict_list: list) -> dict:
        """
        Reducer function: takes a list of partial dicts and merges them into a single dict {word: total_count}.
        """
        final_dict = defaultdict(int)
        for d in dict_list:
            for word, cnt in d.items():
                final_dict[word] += cnt
        return final_dict

    def parallel_mapreduce(self) -> None:
        """
        Perform parallel MapReduce on self.words to compute self.word_counts.
        1. Split words into chunks of size self.chunk_size.
        2. Map each chunk in parallel -> partial dicts.
        3. Reduce partial dicts into one final dict -> self.word_counts.
        """
        logger.info(Fore.CYAN + "Starting parallel MapReduce...")

        # 1. Split into chunks
        chunks = []
        for i in range(0, len(self.words), self.chunk_size):
            chunks.append(self.words[i : i + self.chunk_size])

        # 2. Map step (in parallel)
        partial_dicts = []
        logger.info(
            Fore.YELLOW
            + f"Submitting {len(chunks)} chunks to mapper with {self.max_workers} workers."
        )
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self.mapper, chunk))

            for f in futures:
                partial_dicts.append(f.result())

        logger.info(Fore.YELLOW + "Mapper phase completed. Now reducing...")

        # 3. Reduce step
        self.word_counts = self.reducer(partial_dicts)
        logger.info(
            Fore.GREEN
            + f"Reduce phase completed. Unique words found: {len(self.word_counts)}"
        )

    def visualize_top_words(self) -> None:
        """
        Visualize the top N words from self.word_counts using a horizontal bar chart,
        similar to the provided example image.
        """
        if not self.word_counts:
            logger.warning(
                Fore.RED + "No word counts to visualize. Possibly the text was empty."
            )
            return

        # Sort by frequency (descending), take top_n
        sorted_items = sorted(
            self.word_counts.items(), key=lambda x: x[1], reverse=True
        )
        top_words = sorted_items[: self.top_n]

        words = [item[0] for item in top_words]
        counts = [item[1] for item in top_words]

        plt.figure(figsize=(8, 5))
        plt.barh(words, counts, color="skyblue")
        plt.title(f"Top {self.top_n} Most Frequent Words")
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def run(self) -> None:
        """
        Main orchestrator: fetch text, clean & split, mapreduce, visualize.
        """
        logger.info(Fore.CYAN + "=== Starting WordCountMapReduce process ===")

        # 1) Завантажити текст
        text = self.fetch_text()
        if not text:
            logger.warning(Fore.RED + "No text retrieved. Exiting run().")
            return

        # 2) Розбити на слова
        self.text_clean_and_split(text)

        # 3) Виконати паралельний MapReduce
        self.parallel_mapreduce()

        # 4) Візуалізація
        self.visualize_top_words()


def main():
    """
    Example usage of the WordCountMapReduce class.
    You can adjust URL, chunk_size, max_workers, top_n as desired.
    """
    url = "https://gutenberg.net.au/ebooks01/0100021.txt"

    mapreduce = WordCountMapReduce(url=url, chunk_size=2000, max_workers=4, top_n=10)
    mapreduce.run()


if __name__ == "__main__":
    main()
