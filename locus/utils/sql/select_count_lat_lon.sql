select
    count(id)
from
    dataset
where
    latitude >= {}
    and latitude < {}
    and longitude >= {}
    and longitude < {};