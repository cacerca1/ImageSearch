
class ExtractorSearch():
    def __init__(self) -> None:
        pass

    def get_model(self):
        """Define model. 
        """
        raise NotImplementedError

    def get_embedding(self, enc_img):
        """Get feature embeddings for the given image

        Args:
            enc_img (image): Query image

        """
        raise NotImplementedError
        
    def infer(self, enc_img, k):
        """Search for k similar images to the given image.

        Args:
            enc_img (image): Image to query against.
            k (int): Number of desired similar images.
        """
        raise NotImplementedError
