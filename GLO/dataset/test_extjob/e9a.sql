--- all sci-fi movies with case born in 1990's
SELECT min(t.title), min(pi.info)
FROM person_info as pi, info_type as it1, info_type as it2, name as n, cast_info as ci, title as t, movie_info as mi
WHERE
it2.id = 3
AND mi.info ILIKE '%sci%'
AND it1.info ILIKE 'birth date'
AND pi.info ILIKE '%199%'
AND t.id = mi.movie_id
AND t.id = ci.movie_id
AND mi.movie_id = ci.movie_id
AND mi.info_type_id = it2.id
AND ci.person_id = n.id
AND n.id = pi.person_id
AND pi.info_type_id = it1.id;
