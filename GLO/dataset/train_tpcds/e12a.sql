-- sorted list of top actors from india in given genres
select n.name
from title as t,
name as n,
cast_info as ci,
movie_info as mi,
info_type as it1,
info_type as it2,
person_info as pi
WHERE it1.id = 3
AND (mi.info ILIKE '%romance%'
  OR mi.info ILIKE '%action%')
AND it2.info ILIKE '%birth%'
AND pi.info ILIKE '%usa%'
AND t.id = ci.movie_id
AND t.id = mi.movie_id
AND ci.movie_id = mi.movie_id
AND ci.person_id = n.id
AND it1.id = mi.info_type_id
AND pi.person_id = n.id
AND pi.person_id = ci.person_id
AND pi.info_type_id = it2.id
group by n.name
order by count(*) DESC
