import os
import pypandoc


def convert_word_to_md(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".doc") or filename.endswith(".docx"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".md"
            output_path = os.path.join(output_dir, output_filename)

            try:
                print(f"Converting: {input_path} -> {output_path}")
                pypandoc.convert_file(input_path, "markdown", outputfile=output_path)
                print("Conversion successful!")
            except Exception as e:
                print(f"Error converting {input_path}: {e}")


if __name__ == "__main__":
    input_directory = "./tmp"
    output_directory = "./md_files"

    convert_word_to_md(input_directory, output_directory)
