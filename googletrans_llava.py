from pathlib import Path
import click
import datasets
from googletrans import Translator

DIRECTORY = Path(__file__).parent
translator = Translator()

def donwload_dataset():
    """Download the dataset"""
    url = "https://raw.githubusercontent.com/llava-audio/llava_instruct/main/llava_instruct_150k.json"
    path_dataset = DIRECTORY / "llava_instruct_150k.json"
    datasets.utils.download(url, str(path_dataset))
# Load the dataset
def load_dataset():
    """Load the dataset"""
    path_dataset = DIRECTORY / "llava_instruct_150k.json"
    dataset = datasets.load_dataset("json", data_files=str(path_dataset))
    return dataset

def traslated_dataset(row):
    """Translate the dataset, from English to Spanish"""
    conversations = row["conversations"]
    conversations_translated_obj = translator.translate(conversations, src="en", dest="es")
    conversations_translated = list(map(lambda x: x.text, conversations_translated_obj))
    row["conversations_translated"] = conversations_translated
    return row

@click.command()
@click.option("--chunks_created", default=0, help="Number of chunks created")
def main(chunks_created):
    """Main function"""
    dataset = load_dataset()
    dataset = dataset["train"]
    
    # Define the chunk size (number of examples per chunk)
    chunk_size = 500

    # Calculate the number of chunks needed
    num_chunks = len(dataset) // chunk_size

    # Directory to save the chunks
    output_directory = "dataset"

    # Iterate through the dataset and save each chunk
    print("Total number of chunks: ", num_chunks)
    for i in range(chunks_created, num_chunks):
        print(f"------->Creating chunk {i} / {num_chunks}<-------")
        chunk = dataset[i * chunk_size : (i + 1) * chunk_size]
        chunk = datasets.Dataset.from_dict(chunk)
        dataset_translated = chunk.map(traslated_dataset)
        dataset_translated.to_json(output_directory + f"/chunk_{i}.json")
        chunks_created += 1
        print(f"------->Chunk {i} created<-------")

if __name__ == "__main__":
    main()
    