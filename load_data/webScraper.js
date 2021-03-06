// Script to scrape URLs from WikiArt
// Isaac Krementsov
// 11/17/2020

import fs from 'fs';
import Nightmare from 'nightmare';

// Functions that operate in the browser scope

// Check whether the load button still exists
const checkLoad = () => {
    const button = document.querySelector('a.masonry-load-more-button');
    return button.offsetParent != null;
}
// Check whether all images have loaded (WikiArt uses lazy loading)
const checkLoaded = (selector, i) => {
    let m = i % 2 == 0 ? 1 : -1;
    if(i % 15 == 0) window.scrollTo({top: m*document.body.scrollHeight, behavior: 'smooth'});

    const lazyUrl = 'https://uploads.wikiart.org/Content/wiki/img/lazy-load-placeholder.png';
    const imgs = Array.from(document.querySelectorAll(selector));

    return imgs.every(img => img.src != lazyUrl || !img);
}

const lastResort = (selector, i) => {
    const imgs = Array.from(document.querySelectorAll(selector));
    const lazyUrl = 'https://uploads.wikiart.org/Content/wiki/img/lazy-load-placeholder.png';
    const notLoaded = imgs.filter(img => img.src == lazyUrl);

    if(notLoaded.length > 1){
        notLoaded[0].scrollIntoView({behavior: 'smooth'});
        return false;
    }else{
        return true;
    }
}

// Get all image srcs on the page
const getImages = selector => {
    let imgs = document.querySelectorAll(selector);
    let imgArr = Array.from(imgs);

    return imgArr.map(img => img.src.split('!')[0]);
}

// Use Nightmare to scrape painting URLs from a genre
async function scrapeGenre(genre){

    let loadMoreExists = true;
    let j = 0;

    // Keep clicking load more while it's an option
    while(loadMoreExists){
        try {
            await genre.click('a.masonry-load-more-button');
            loadMoreExists = await genre.evaluate(checkLoad);
        }catch(_e){
            break;
        }

        j++;
    }

    let lastImageLoaded = false;
    let i = 0;

    // Wait for lazy loading to finish
    while(!lastImageLoaded){
        await genre.wait(1000);

        if(i <= 90){
            lastImageLoaded = await genre.evaluate(checkLoaded, 'li.ng-scope img', i);
        }else if(i % 3 == 0){
            lastImageLoaded = await genre.evaluate(lastResort, 'li.ng-scope img', i);
        }
        i++;
    }

    // Get a list of scraped image URLs and add to ./urls.json
    const image_srcs = {urls: await genre.evaluate(getImages, 'li.ng-scope img')};

    fs.writeFileSync('./urls.json', JSON.stringify(image_srcs));
}

// Get data from the abstract art section
const url = 'https://www.wikiart.org/en/paintings-by-genre/abstract?select=featured#!#filterName:featured,viewType:masonry';

const nightmare = Nightmare({show: true});
const genre = nightmare.goto(url);

const start = new Date().getTime();

scrapeGenre(genre).then(() => {
    // Report how long the scraping took
    const elapsed = new Date().getTime() - start;
    console.log('Data scraped in', Math.round(elapsed/10)/100, 'seconds');

    // End NightMare and Node
    genre.end();
    process.exit();
});
