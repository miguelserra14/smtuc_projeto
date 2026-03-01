# MVP — Alocação eficiente de autocarros e metrobus

visa responder a questao: agora que o metrobus faz parte da mobilidade em coimbra, como adaptar os smtuc a esta mudanca de forma a cobrir as suas mais conhecidas e habituais lacunas, assim como nao so se adaptar a coimbra de hoje mas a de amanha

## to do :
- dados:
- corrigir e limpar dados metrobus: locais das paragens, corrigir horarios, arranjar shape.txt
- atualizar dados smtuc: pq e que da erro no dados.gov?

- gtfs: colocar numa pasta separada? tentar tornar mais pequeno e flexivel

- operations: separar em 2 (route_operations e overlap_operations)
- brincar mais com os dados: sera que o output está certo sequer? como é que ele chegou a estes valores mm? (mt importante na do overlap) 
- fazer com que seja mais facil testar eustoes como:
- como ir de ponto a a b no horario x?
- quais as paragens mais proximas de um ponto a e b?
- qual o alcance de uma paragem em 15min em varios horarios
- vista por linha: frequencia, overlap com o metrobus
- quais as areas c menos overlap? e dessas quais e que se encontram mais no centro da cidade? e quais têm pior frequência?
- começar a pensasr em partes para criar um score final de necessidade com base em população, frequencia, overlap ou falta dele

- visualização
- meter tudo num mapa : linhas, paragens, so smtuc, so metrobus
- conseguir visualizar cada linha e comparar overlap com o metrobus, dar estatisticas em percentagem
- limpar slop: evitar duplicados e fazer com que seja mais facil testar eustoes como:
- como ir de ponto a a b no horario x?
- quais as paragens mais proximas de um ponto a e b?
- qual o alcance de uma paragem em 15min em varios horarios
- vista por linha: frequencia, overlap com o metrobus


# MVP — Alocação eficiente de autocarros e metrobus (Coimbra)

## Enquadramento

Este projeto procura responder à pergunta:

**Agora que o Metrobus faz parte da mobilidade em Coimbra, como adaptar os SMTUC para cobrir lacunas atuais e preparar a cidade para necessidades futuras?**

## Objetivos do MVP

- Integrar e validar dados GTFS de **SMTUC** e **Metrobus**.
- Comparar cobertura e oferta entre as duas redes.
- Testar trajetos reais (A → B) por dia e hora.
- Apoiar decisões de ajuste de linhas e frequências.

## To-Do

### 1: Dados

- [ ] Corrigir e limpar dados do Metrobus:
  - localização das paragens;
  - consistência de horários.
- [ ] Atualizar dados SMTUC:
  - investigar erro no `dados.gov`.
- [ ] Validar consistência GTFS:
  - `stops`, `trips`, `stop_times`, `calendar`, `calendar_dates`.

### 2: Probe (análise)

- [ ] Limpar ruído e evitar duplicados nos resultados.
- [ ] Facilitar testes de perguntas como:
  - como ir de ponto A para B no horário X?
  - quais as paragens mais próximas de A e de B?
  - qual o alcance de uma paragem em 15 min em diferentes horários?
  - vista por linha: frequência e overlap com Metrobus.
- [ ] Melhorar pesquisa para incluir transbordos (além de trajetos diretos).

### 3: Visualização

- [ ] Colocar tudo num mapa:
  - linhas;
  - paragens;
  - filtro SMTUC;
  - filtro Metrobus.
- [ ] Comparar cada linha com overlap Metrobus.
- [ ] Mostrar estatísticas percentuais por linha/corredor.

## Resultado esperado

No fim do MVP, deve ser possível responder com dados a:

- onde o Metrobus já cobre bem a procura;
- onde os SMTUC devem reforçar/adaptar serviço;
- que alterações melhoram tempo de viagem e cobertura.