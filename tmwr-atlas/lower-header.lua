function Header(el)
  -- The header level can be accessed via the attribute 'level'
  -- of the element. See the Pandoc documentation later.
  if (el.level <= 1) then
    return el
  end
  el.level = el.level + 1
  return el
end
