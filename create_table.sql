create table dataset
(
    id        int generated always as identity
        constraint dataset_pk
            primary key,
    latitude  real  not null,
    longitude real  not null,
    image     bytea not null
);

comment on table dataset is 'Table for storing all the images in the LDoGI dataset formatted for easy access.';

comment on column dataset.id is 'Datapoint id';

comment on column dataset.latitude is 'Latitude of the coordinate where the picture was taken.';

comment on column dataset.longitude is 'Longitude of the coordinate where the picture was taken.';

comment on column dataset.image is 'Image data in binary format.';
