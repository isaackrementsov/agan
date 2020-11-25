import fs from 'fs';
import path from 'path';
import Calipers from 'calipers';
import { promisify } from 'util';

const __dirname = path.resolve();
const raw = path.join(__dirname, '../raw/');

const calipers = Calipers('jpeg', 'png');
const measure = promisify(calipers.measure);

fs.readdir(raw, async (err, files) => {
    if(err){
        console.log(err);
    }else{
        let data = JSON.parse(fs.readFileSync('urls.json'));
        let urls = data.urls;
        let minMax = 200;
        let minMin = 200;

        try {
            for(let file of files){
                let result = await calipers.measure(path.join(raw, file));
                let dimensions = result.pages[0];

                let dimMin = Math.min(dimensions.height, dimensions.width);
                let dimMax = Math.max(dimensions.height, dimensions.width);

                if(dimMax < minMax || dimMin < minMin){
                    let url = urls.find(u => {
                        let parts = u.split('/');
                        let l = parts.length - 1;
                        return (parts[l - 1] + parts[l]) == file;
                    });
                    console.log(url);
                }
            }
        }catch(e){
            console.log(e);
        }
    }
});
