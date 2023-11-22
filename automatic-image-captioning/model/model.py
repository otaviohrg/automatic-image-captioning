import torch
import torch.nn as nn


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
                output = self.decoder.linear(hidden.unsqueeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.index_to_string[predicted.item()] == "<EOS>":
                    break

            return [vocabulary.index_to_string[index] for index in result_caption]
