import apache_beam as beam

class ImageToShard(beam.DoFn):

    def __init__(self, class_to_idx):
        self.class_to_idx = class_to_idx

                              
    def process(self, element):
        # Implement your logic to convert the image to a shard
        # For example, you can read the image, resize it, and save it as a shard
        # it will receive file path and yield bytes
        with open(element, 'rb') as f:
            image_bytes = f.read()
        yield image_bytes

def run_conversion():
    # first collect all the image paths in the dataset
    dataset_root = "/gcs/caltech_1000/"
    output_prefix = "/gcs/caltech_1000_shards/shard"
      
    image_paths = []
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                
                # Get the immediate parent folder name
                folder_name = os.path.basename(root)
                if folder_name and folder_name != "caltech_1000":
                    categories.add(folder_name)

        # Generate the class index map cleanly
    class_to_idx = {cat: i for i, cat in enumerate(sorted(categories))}
    print(f"Found {len(image_paths)} images across {len(class_to_idx)} classes.")


    with beam.Pipeline() as pipeline:
        # write as python shard 
        gcs = beam.io.gcp.gcsfilesystem.GCSFileSystem()
        pipeline | "create image paths" >> beam.Create(image_paths) \
                    | "Convert to Shards" >> beam.ParDo(ImageToShard(class_to_idx)) \
                    | "Write Shards" >> beam.io.WriteToTFRecord("/gcs/caltech_1000_shards/shard", file_name_suffix=".tfrecord")    
       