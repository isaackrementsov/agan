## aGAN
Generative Adversarial Network to visualize physical systems with abstract art

#### Install dependencies

```bash
pip3 install -r requirements.txt
```

#### Run tests
Generate a GIF & image from random inputs
```bash
python3 test_generator.py
```
Generate an image from sample physical data
```bash
python3 test.py
```

#### Training
##### Train the generator (do this first)
```bash
python3 train_generator.py
```
##### Train the latent space mapper (do this after training the generator)
```bash
python3 train_mapper.py
```
