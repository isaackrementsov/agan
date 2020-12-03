import fs from 'fs';
import path from 'path';
import sharp from 'sharp';

const __dirname = path.resolve();
const raw = path.join(__dirname, '../raw/');
const assets = path.join(__dirname, '../assets');

fs.readdir(raw, (err, files) => {
    if(err){
        console.log(err);
    }else{
        let dim = 450;

        for(let file of files){
            sharp(path.join(raw, file)).resize(dim, dim).toFile(path.join(assets, file));
        }
    }
});
