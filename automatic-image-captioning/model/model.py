import torch
import torch.nn as nn

from .decoder.lstm import LSTMDecoder
from .encoder.inception import InceptionEncoder


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_model, decoder_model, embed_size, hidden_size, vocab_size, num_layers):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder_model(embed_size)
        self.decoder = decoder_model(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hidden, states = self.decoder.model(x, states)
                output = self.decoder.linear(hidden.squeeze(0))
                predicted = output.argmax(0)

                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.index_to_string[predicted.item()] == "<EOS>":
                    break

            return [vocabulary.index_to_string[index] for index in result_caption]


if __name__ == "__main__":
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    num_epochs = 100
    vocab_size = 100

    model = EncoderDecoder(InceptionEncoder, LSTMDecoder, embed_size, hidden_size, vocab_size, num_layers)

    print(model)
