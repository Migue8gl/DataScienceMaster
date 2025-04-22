import csv
import re
import unicodedata


def clean_word(word):
    # Replace underscores with spaces
    word = word.replace("_", " ")
    # Remove accents
    word = unicodedata.normalize("NFD", word)
    word = word.encode("ascii", "ignore").decode("utf-8")
    # Convert to lowercase
    word = word.lower()
    return word


def process_sentiwordnet_file(input_path, output_path):
    seen_words = set()
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["word", "polarity", "score"])

        with open(input_path, "r", encoding="utf-8") as infile:
            for line in infile:
                if line.startswith("#") or line.strip() == "":
                    continue

                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    pos, id_num, pos_score, neg_score, synset_terms, *gloss_parts = (
                        parts
                    )
                    pos_score = float(pos_score)
                    neg_score = float(neg_score)

                    if pos_score > 0 or neg_score > 0:
                        words = synset_terms.split()
                        for word in words:
                            word_clean = word.split("#")[0]
                            word_clean = clean_word(word_clean)

                            if word_clean not in seen_words:
                                seen_words.add(word_clean)
                                if pos_score >= neg_score:
                                    csv_writer.writerow(
                                        [word_clean, "positive", pos_score]
                                    )
                                else:
                                    csv_writer.writerow(
                                        [word_clean, "negative", neg_score]
                                    )


def process_subjclue_file(input_path, output_path):
    seen_words = set()
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["word", "polarity"])

        with open(input_path, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                # Extract word and polarity from line
                word_match = re.search(r"word1=(\w+)", line)
                polarity_match = re.search(r"priorpolarity=(\w+)", line)

                if word_match and polarity_match:
                    word = word_match.group(1)
                    polarity = polarity_match.group(1)

                    word_clean = clean_word(word)

                    if word_clean not in seen_words:
                        seen_words.add(word_clean)
                        csv_writer.writerow([word_clean, polarity])


def process_senticnet_file(input_path, output_path):
    seen_words = set()
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["word", "polarity", "score"])

        with open(input_path, "r", encoding="utf-8") as infile:
            # Skip the header line
            header = next(infile, None)

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                # Use regex to split the line into parts, handling variable whitespace
                parts = re.split(r"\s{2,}|\t+", line)

                if len(parts) >= 3:
                    concept = parts[0].strip()
                    polarity = parts[1].strip().lower()
                    intensity = parts[2].strip()

                    # Clean the concept word
                    word_clean = clean_word(concept)

                    # Convert intensity to float
                    try:
                        intensity_float = float(intensity)

                        if word_clean not in seen_words:
                            seen_words.add(word_clean)
                            csv_writer.writerow([word_clean, polarity, intensity_float])
                    except ValueError:
                        print(
                            f"Skipping invalid intensity value: {intensity} for word: {word_clean}"
                        )


if __name__ == "__main__":
    # Process SentiWordNet file
    sentiwordnet_input = "Análisis de sentimientos con KNIME-20250421/Lexicon/SentiWordNet 3.0.0/SentiWordNet_3.0.0_20130122.txt"
    sentiwordnet_output = "sentiwordnet.csv"
    process_sentiwordnet_file(sentiwordnet_input, sentiwordnet_output)

    # Process SubjClue file
    subjclue_input = "subjclueslen1-HLTEMNLP05.tff"
    subjclue_output = "subjclue.csv"
    process_subjclue_file(subjclue_input, subjclue_output)

    # Process SenticNet5 file
    senticnet_input = "senticnet5.txt"
    senticnet_output = "senticnet.csv"
    process_senticnet_file(senticnet_input, senticnet_output)
